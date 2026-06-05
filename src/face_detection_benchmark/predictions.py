"""Shared prediction records for benchmark model outputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from face_detection_benchmark.config import FACE_CATEGORY_NAME


@dataclass(frozen=True)
class DetectionRecord:
    """One normalized face detection from any benchmark model."""

    bbox_xyxy: list[float]
    bbox_xywh: list[float]
    confidence: float | None
    class_id: int | None
    class_name: str = FACE_CATEGORY_NAME


@dataclass(frozen=True)
class ImagePredictionRecord:
    """Normalized predictions for one benchmark image."""

    file_name: str
    image_path: str
    width: int
    height: int
    detections: list[DetectionRecord]
    model_name: str
    model_config: dict[str, Any] = field(default_factory=dict)
    source_video: str | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None
    threshold: float | None = None
    device: str | None = None
    backend: str | None = None
    timing_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResult:
    """Summary of a prediction run."""

    output_path: Path
    image_count: int
    detection_count: int
    preview_dir: Path | None
    preview_count: int
    latency_path: Path | None = None
    latency_table_path: Path | None = None


def prediction_record_to_json(record: ImagePredictionRecord) -> dict[str, Any]:
    """Convert a prediction record to a JSON-serializable dictionary."""
    return {
        **asdict(record),
        "detections": [asdict(detection) for detection in record.detections],
    }


def summarize_latency(
    *,
    model_name: str,
    backend: str,
    device: str,
    image_count: int,
    detection_count: int,
    inference_times_ms: list[float],
    total_runtime_ms: float,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize per-image inference latency for a benchmark prediction run."""
    if image_count != len(inference_times_ms):
        raise ValueError(
            "Latency count must match image count "
            f"({len(inference_times_ms)} != {image_count})"
        )
    sorted_times = sorted(inference_times_ms)
    return {
        "model_name": model_name,
        "backend": backend,
        "device": device,
        "image_count": image_count,
        "detection_count": detection_count,
        "total_runtime_ms": round(total_runtime_ms, 4),
        "total_inference_ms": round(sum(inference_times_ms), 4),
        "per_image_inference_ms": {
            "mean": round(sum(inference_times_ms) / image_count, 4)
            if image_count
            else 0.0,
            "median": round(_percentile(sorted_times, 50), 4),
            "p90": round(_percentile(sorted_times, 90), 4),
            "min": round(sorted_times[0], 4) if sorted_times else 0.0,
            "max": round(sorted_times[-1], 4) if sorted_times else 0.0,
        },
        "model_config": model_config,
    }


def write_latency_summary(
    *,
    summary: dict[str, Any],
    latency_path: Path,
    latency_table_path: Path | None = None,
) -> None:
    """Write benchmark latency artifacts as JSON and optionally append CSV."""
    latency_path.parent.mkdir(parents=True, exist_ok=True)
    latency_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if latency_table_path is not None:
        latency_table_path.parent.mkdir(parents=True, exist_ok=True)
        _append_latency_csv_row(summary, latency_table_path)


def xyxy_to_xywh(bbox_xyxy: list[float]) -> list[float]:
    """Convert an xyxy box to xywh format with non-negative dimensions."""
    x1, y1, x2, y2 = bbox_xyxy
    return [
        round(x1, 4),
        round(y1, 4),
        round(max(0.0, x2 - x1), 4),
        round(max(0.0, y2 - y1), 4),
    ]


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def _append_latency_csv_row(summary: dict[str, Any], latency_table_path: Path) -> None:
    fieldnames = [
        "model_name",
        "backend",
        "device",
        "image_count",
        "detection_count",
        "total_runtime_ms",
        "total_inference_ms",
        "mean_inference_ms",
        "median_inference_ms",
        "p90_inference_ms",
        "min_inference_ms",
        "max_inference_ms",
    ]
    write_header = not latency_table_path.exists()
    with latency_table_path.open("a", encoding="utf-8", newline="") as latency_file:
        writer = csv.DictWriter(latency_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        per_image = summary["per_image_inference_ms"]
        writer.writerow(
            {
                "model_name": summary["model_name"],
                "backend": summary["backend"],
                "device": summary["device"],
                "image_count": summary["image_count"],
                "detection_count": summary["detection_count"],
                "total_runtime_ms": summary["total_runtime_ms"],
                "total_inference_ms": summary["total_inference_ms"],
                "mean_inference_ms": per_image["mean"],
                "median_inference_ms": per_image["median"],
                "p90_inference_ms": per_image["p90"],
                "min_inference_ms": per_image["min"],
                "max_inference_ms": per_image["max"],
            }
        )
