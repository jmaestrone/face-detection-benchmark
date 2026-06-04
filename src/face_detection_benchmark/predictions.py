"""Shared prediction records for benchmark model outputs."""

from __future__ import annotations

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


def prediction_record_to_json(record: ImagePredictionRecord) -> dict[str, Any]:
    """Convert a prediction record to a JSON-serializable dictionary."""
    return {
        **asdict(record),
        "detections": [asdict(detection) for detection in record.detections],
    }


def xyxy_to_xywh(bbox_xyxy: list[float]) -> list[float]:
    """Convert an xyxy box to xywh format with non-negative dimensions."""
    x1, y1, x2, y2 = bbox_xyxy
    return [
        round(x1, 4),
        round(y1, 4),
        round(max(0.0, x2 - x1), 4),
        round(max(0.0, y2 - y1), 4),
    ]
