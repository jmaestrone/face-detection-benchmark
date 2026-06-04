"""Small environment helpers for local, private benchmark configuration."""

from __future__ import annotations

import os
from pathlib import Path


def get_env_value(name: str, env_path: Path = Path(".env")) -> str | None:
    """Return an environment value, falling back to a local .env file."""
    value = os.environ.get(name)
    if value:
        return value
    return read_dotenv_values(env_path).get(name)


def read_dotenv_values(env_path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE lines from a local .env file."""
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        value = raw_value.strip().strip("'\"")
        if key:
            values[key] = value
    return values
