"""Central configuration for AI Career Copilot.

Values are read from the environment (optionally via a local ``.env`` file).
Everything has a sensible default so the app runs with zero configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # python-dotenv is optional; the app works without it.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - trivial branch
    pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings resolved from the environment."""

    sbert_model: str = os.getenv("SBERT_MODEL", "all-MiniLM-L6-v2")

    adzuna_app_id: str | None = os.getenv("ADZUNA_APP_ID") or None
    adzuna_app_key: str | None = os.getenv("ADZUNA_APP_KEY") or None
    adzuna_country: str = os.getenv("ADZUNA_COUNTRY", "gb")

    request_timeout: int = _env_int("REQUEST_TIMEOUT", 15)

    # Result-set sizing defaults.
    default_top_k: int = 10
    max_jobs: int = 60

    # Providers that require no credentials are on by default.
    default_providers: tuple[str, ...] = field(
        default_factory=lambda: ("remotive", "arbeitnow")
    )

    @property
    def adzuna_enabled(self) -> bool:
        return bool(self.adzuna_app_id and self.adzuna_app_key)


settings = Settings()
