"""AI Career Copilot — Streamlit application entry point.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import settings
from src.jobs.arbeitnow import ArbeitnowProvider
from src.jobs.adzuna import AdzunaProvider
from src.jobs.remotive import RemotiveProvider
from src.jobs.service import JobSearchService, PROVIDER_REGISTRY
from src.matching.embeddings import EmbedderUnavailableError, build_embedder
from src.matching.ranker import rank_from_scores, score_jobs
from src.matching.skill_gap import aggregate_demand
from src.models import SearchFilters
from src.parser.pdf import PdfExtractionError, extract_text_from_pdf
from src.parser.resume_parser import ResumeParser
from src.skills.taxonomy import default_taxonomy
from src.ui import components as ui

st.set_page_config(page_title="AI Career Copilot", page_icon="🧭", layout="wide")

EMBEDDER_CHOICES = {
    "Sentence-BERT (semantic)": "auto",
    "TF-IDF (lexical fallback)": "tfidf",
}
EXPERIENCE_CHOICES = ["Any", "Entry", "Mid", "Senior"]


@st.cache_resource(show_spinner=False)
def get_taxonomy():
    return default_taxonomy()


@st.cache_resource(show_spinner=False)
def get_embedder(preference: str, model_name: str):
    return build_embedder(preference, model_name)


def build_service(provider_names, adzuna_creds) -> JobSearchService:
    providers = []
    for name in provider_names:
        if name == RemotiveProvider.name:
            providers.append(RemotiveProvider(timeout=settings.request_timeout))
        elif name == ArbeitnowProvider.name:
            providers.append(ArbeitnowProvider(timeout=settings.request_timeout))
        elif name == AdzunaProvider.name and adzuna_creds:
            app_id, app_key = adzuna_creds
            providers.append(
                AdzunaProvider(
                    app_id, app_key, settings.adzuna_country, settings.request_timeout
                )
            )
    return JobSearchService(providers)


def read_resume(uploaded_file, pasted_text: str) -> str:
    """Return resume text from an uploaded PDF or the paste box."""
    if uploaded_file is not None:
        return extract_text_from_pdf(uploaded_file.getvalue())
    return pasted_text or ""


def render_sidebar():
    st.sidebar.title("🧭 AI Career Copilot")
    st.sidebar.caption("Resume-based job matching with semantic search and skill-gap analysis.")

    st.sidebar.header("1 · Your resume")
    uploaded = st.sidebar.file_uploader("Upload a PDF resume", type=["pdf"])
    pasted = st.sidebar.text_area(
        "…or paste resume text", height=140, placeholder="Paste your resume here"
    )

    st.sidebar.header("2 · Job search")
    query = st.sidebar.text_input(
        "Target role / keywords", placeholder="e.g. machine learning engineer"
    )
    location = st.sidebar.text_input("Location", placeholder="e.g. London (optional)")
    col_a, col_b = st.sidebar.columns(2)
    experience = col_a.selectbox("Experience", EXPERIENCE_CHOICES, index=0)
    employment_type = col_b.text_input("Type", placeholder="full-time")
    remote_only = st.sidebar.checkbox("Remote only", value=False)

    st.sidebar.header("3 · Providers")
    provider_options = list(PROVIDER_REGISTRY.keys())
    if settings.adzuna_enabled:
        provider_options.append(AdzunaProvider.name)
    providers = st.sidebar.multiselect(
        "Job sources", provider_options, default=provider_options[:2]
    )
    if AdzunaProvider.name in provider_options and AdzunaProvider.name not in providers:
        st.sidebar.caption("Adzuna credentials detected — enable it above to include it.")

    st.sidebar.header("4 · Matching")
    embedder_label = st.sidebar.selectbox("Embedding model", list(EMBEDDER_CHOICES), index=0)
    top_k = st.sidebar.slider("Jobs to return", 3, 30, 10)
    max_jobs = st.sidebar.slider("Candidates to retrieve", 10, settings.max_jobs, 40, step=5)
    compare = st.sidebar.checkbox("Compare with keyword baseline", value=True)

    return {
        "uploaded": uploaded,
        "pasted": pasted,
        "filters": SearchFilters(
            query=query,
            location=location or None,
            experience_level=experience,
            employment_type=employment_type or None,
            remote_only=remote_only,
        ),
        "providers": providers,
        "embedder_pref": EMBEDDER_CHOICES[embedder_label],
        "top_k": top_k,
        "max_jobs": max_jobs,
        "compare": compare,
    }


def run_analysis(options):
    resume_text = read_resume(options["uploaded"], options["pasted"])
    if not resume_text.strip():
        st.error("Please upload a PDF resume or paste your resume text.")
        return None
    if not options["filters"].query.strip():
        st.error("Please enter a target role or keywords to search for.")
        return None

    taxonomy = get_taxonomy()
    parser = ResumeParser(taxonomy)
    resume = parser.parse_text(resume_text)

    embedder, warning = get_embedder(options["embedder_pref"], settings.sbert_model)

    with st.spinner("Retrieving jobs…"):
        service = build_service(options["providers"], (
            (settings.adzuna_app_id, settings.adzuna_app_key)
            if settings.adzuna_enabled
            else None
        ))
        report = service.search(options["filters"], limit=options["max_jobs"])

    if not report.jobs:
        st.session_state["analysis"] = {
            "resume": resume,
            "warning": warning,
            "report": report,
            "results": [],
            "baseline": [],
        }
        return st.session_state["analysis"]

    with st.spinner("Ranking and analysing skill gaps…"):
        scores = score_jobs(resume, report.jobs, embedder=embedder, method="semantic")
        results = rank_from_scores(
            resume, report.jobs, scores, taxonomy=taxonomy, top_k=options["top_k"]
        )
        baseline = []
        if options["compare"]:
            baseline_scores = score_jobs(resume, report.jobs, method="keyword")
            baseline = rank_from_scores(
                resume, report.jobs, baseline_scores, taxonomy=taxonomy
            )

    analysis = {
        "resume": resume,
        "warning": warning,
        "report": report,
        "results": results,
        "baseline": baseline,
    }
    st.session_state["analysis"] = analysis
    return analysis


def render_analysis(analysis, compare: bool):
    resume = analysis["resume"]
    report = analysis["report"]
    results = analysis["results"]

    st.subheader("Parsed profile")
    ui.render_profile(resume, analysis.get("warning"))

    if report.errors:
        for provider, error in report.errors.items():
            st.warning(f"Provider '{provider}' failed: {error}")

    if not results:
        st.info(
            f"No jobs matched your filters (retrieved {report.raw_count} raw listings). "
            "Try broader keywords, fewer filters, or additional providers."
        )
        return

    st.subheader("Recommendations")
    ui.render_overview(results)

    st.subheader("Skill-gap insights")
    ui.render_skill_gap_summary(results, aggregate_demand(results, top_n=15))

    st.subheader("Ranked matches")
    for result in results:
        ui.render_job_card(result, result.rank)

    if compare and analysis.get("baseline"):
        st.subheader("Semantic vs keyword ranking")
        ui.render_comparison(results, analysis["baseline"])

    st.download_button(
        "⬇️ Download results as CSV",
        data=ui.results_to_dataframe(results).to_csv(index=False).encode("utf-8"),
        file_name="career_copilot_matches.csv",
        mime="text/csv",
    )


def main():
    options = render_sidebar()

    st.title("AI Career Copilot")
    st.caption(
        "Semantic resume–job matching with Sentence-BERT, relevance ranking and "
        "skill-gap analysis. No database: all processing happens in memory for the "
        "active session."
    )

    if st.button("🔍 Find matching jobs", type="primary"):
        with st.spinner("Parsing resume…"):
            try:
                run_analysis(options)
            except PdfExtractionError as exc:
                st.error(f"Could not read the resume PDF: {exc}")
            except EmbedderUnavailableError as exc:
                st.error(str(exc))

    analysis = st.session_state.get("analysis")
    if analysis:
        st.divider()
        render_analysis(analysis, options["compare"])
    else:
        st.info("Upload a resume, set your filters on the left, then click **Find matching jobs**.")


if __name__ == "__main__":
    main()
