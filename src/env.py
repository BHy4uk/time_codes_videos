from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


def load_env() -> None:
    """Load environment variables from a local .env if present."""

    # Load from current working directory .env first
    if load_dotenv is not None:
        load_dotenv(override=False)


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def get_env_path(name: str) -> Path | None:
    val = os.getenv(name)
    return Path(val) if val else None
