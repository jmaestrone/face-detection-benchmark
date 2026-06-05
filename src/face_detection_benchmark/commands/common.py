"""Shared helpers for CLI command implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from face_detection_benchmark.env import get_env_value


def default_run_id() -> str:
    """Return a UTC timestamp suitable for run output directories."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def optional_int_env(name: str) -> int | None:
    """Read an optional integer environment value."""
    value = get_env_value(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def benchmark_latency_paths(
    predictions_path: Path,
    model_name: str,
) -> tuple[Path, Path]:
    """Return latency JSON and CSV paths for a benchmark prediction output."""
    run_dir = (
        predictions_path.parent.parent
        if predictions_path.parent.name == "predictions"
        else predictions_path.parent
    )
    return run_dir / "latency" / f"{model_name}.json", run_dir / "latency.csv"


def parse_thresholds(value: str) -> tuple[float, ...]:
    """Parse comma-separated confidence thresholds from a CLI option."""
    threshold_parts = [part.strip() for part in value.split(",")]
    if not threshold_parts or any(not part for part in threshold_parts):
        raise ValueError("--thresholds must be a comma-separated list of numbers")
    try:
        return tuple(float(part) for part in threshold_parts)
    except ValueError as error:
        raise ValueError("--thresholds must contain only numbers") from error
