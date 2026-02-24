from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from a local .env if present."""

    # Load from current working directory .env first
    load_dotenv(override=False)


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def get_env_path(name: str) -> Path | None:
    val = os.getenv(name)
    return Path(val) if val else None
