"""RF-DETR model adapter for normalized face detection predictions."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    FACE_CATEGORY_NAME,
)
from face_detection_benchmark.predictions import DetectionRecord, xyxy_to_xywh

RFDETR_DATASET_FILES = ("roboflow", "coco", "yolo", "o365")


@dataclass(frozen=True)
class RfdetrConfig:
    """Configuration used to load and run the RF-DETR face detector."""

    weights_path: Path
    device: str
    threshold: float
    max_detections: int


class RfdetrFaceDetector:
    """Adapter that converts RF-DETR predictions into normalized detections."""

    model_name = "rfdetr"
    backend = "rfdetr"

    def __init__(self, config: RfdetrConfig) -> None:
        """Initialize the RF-DETR model adapter from configuration."""
        self.config = config
        self.model = self._load_model(config)

    def predict_batch(self, images_rgb: list[Any]) -> list[list[DetectionRecord]]:
        """Run RF-DETR on RGB images and return normalized detections."""
        predictions = self.model.predict(images_rgb, threshold=self.config.threshold)
        if not isinstance(predictions, list):
            predictions = [predictions]
        if len(predictions) != len(images_rgb):
            raise ValueError(
                "RF-DETR returned a different number of prediction results "
                f"({len(predictions)}) than input images ({len(images_rgb)})"
            )
        return [self._detections_to_records(prediction) for prediction in predictions]

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable model configuration metadata."""
        return {
            "weights_path": self.config.weights_path.as_posix(),
            "checkpoint_name": self.config.weights_path.name,
            "device": self.config.device,
            "threshold": self.config.threshold,
            "max_detections": self.config.max_detections,
        }

    def _load_model(self, config: RfdetrConfig) -> Any:
        """Load the configured RF-DETR model implementation."""
        from rfdetr import RFDETRLarge

        return RFDETRLarge(
            pretrain_weights=str(config.weights_path),
            device=config.device,
            num_classes=2,
            num_queries=config.max_detections,
            num_select=config.max_detections,
        )

    def _detections_to_records(self, prediction: Any) -> list[DetectionRecord]:
        """Convert one RF-DETR prediction object into detection records."""
        xyxy_values = getattr(prediction, "xyxy", [])
        confidence_values = getattr(prediction, "confidence", None)
        class_id_values = getattr(prediction, "class_id", None)

        records: list[DetectionRecord] = []
        for index, xyxy in enumerate(xyxy_values):
            bbox_xyxy = [round(float(value), 4) for value in xyxy]
            records.append(
                DetectionRecord(
                    bbox_xyxy=bbox_xyxy,
                    bbox_xywh=xyxy_to_xywh(bbox_xyxy),
                    confidence=_optional_float(confidence_values, index),
                    class_id=_optional_int(class_id_values, index),
                    class_name=FACE_CATEGORY_NAME,
                )
            )
        return records


def _optional_float(values: Any, index: int) -> float | None:
    """Read an optional float value from a vector-like prediction field."""
    if values is None:
        return None
    return round(float(values[index]), 6)


def _optional_int(values: Any, index: int) -> int | None:
    """Read an optional integer value from a vector-like prediction field."""
    if values is None:
        return None
    return int(values[index])


@dataclass(frozen=True)
class RfdetrTrainingConfig:
    """Configuration used to train an RF-DETR model."""

    dataset_dir: Path
    output_dir: Path
    epochs: int
    batch_size: int
    device: str
    dataset_file: str = "roboflow"
    num_workers: int = 2
    weights_path: Path | None = None


@dataclass(frozen=True)
class RfdetrTrainingResult:
    """Paths written by an RF-DETR training run."""

    output_dir: Path
    config_path: Path
    metadata_path: Path


def train_rfdetr(config: RfdetrTrainingConfig) -> RfdetrTrainingResult:
    """Train RF-DETR with validated local dataset guardrails."""
    _validate_training_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    config_path = config.output_dir / "config.json"
    metadata_path = config.output_dir / "metadata.json"
    started_at = datetime.now(timezone.utc).isoformat()
    config_payload = _training_config_payload(config)
    metadata_payload = {
        "started_at": started_at,
        "status": "started",
        "package_versions": {"rfdetr": importlib.metadata.version("rfdetr")},
    }
    _write_json(config_path, config_payload)
    _write_json(metadata_path, metadata_payload)

    from rfdetr import RFDETRLarge

    model_kwargs: dict[str, Any] = {}
    if config.weights_path is not None:
        model_kwargs["pretrain_weights"] = str(config.weights_path)
    model = RFDETRLarge(**model_kwargs)
    try:
        model.train(
            dataset_dir=str(config.dataset_dir),
            output_dir=str(config.output_dir),
            epochs=config.epochs,
            batch_size=config.batch_size,
            device=config.device,
            dataset_file=config.dataset_file,
            num_workers=config.num_workers,
            notes=config_payload,
        )
    except ImportError as error:
        raise ValueError(str(error)) from error

    metadata_payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata_payload["status"] = "completed"
    _write_json(metadata_path, metadata_payload)
    return RfdetrTrainingResult(
        output_dir=config.output_dir,
        config_path=config_path,
        metadata_path=metadata_path,
    )


def _validate_training_config(config: RfdetrTrainingConfig) -> None:
    """Validate RF-DETR training options before loading the model."""
    if _path_is_inside(config.dataset_dir, DEFAULT_BENCHMARK_DATA_DIR):
        raise ValueError(
            "--dataset-dir must not point inside data/benchmark; use data/training/ "
            "for RF-DETR training datasets"
        )
    if not config.dataset_dir.exists():
        raise ValueError(
            f"Training dataset directory does not exist: {config.dataset_dir}"
        )
    if not config.dataset_dir.is_dir():
        raise ValueError(
            f"Training dataset path is not a directory: {config.dataset_dir}"
        )
    if config.weights_path is not None and not config.weights_path.exists():
        raise ValueError(f"RF-DETR weights file does not exist: {config.weights_path}")
    if config.epochs <= 0:
        raise ValueError("--epochs must be greater than 0")
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if config.num_workers < 0:
        raise ValueError("--num-workers must be greater than or equal to 0")
    if config.device not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("--device must be one of: auto, mps, cuda, cpu")
    if config.dataset_file not in RFDETR_DATASET_FILES:
        allowed = ", ".join(RFDETR_DATASET_FILES)
        raise ValueError(f"--dataset-file must be one of: {allowed}")


def _path_is_inside(candidate: Path, root: Path) -> bool:
    """Return whether candidate points at or under root."""
    resolved_candidate = candidate.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return (
        resolved_candidate == resolved_root
        or resolved_root in resolved_candidate.parents
    )


def _training_config_payload(config: RfdetrTrainingConfig) -> dict[str, Any]:
    """Return JSON-serializable RF-DETR training configuration."""
    return {
        "dataset_dir": config.dataset_dir.as_posix(),
        "output_dir": config.output_dir.as_posix(),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "device": config.device,
        "dataset_file": config.dataset_file,
        "num_workers": config.num_workers,
        "weights_path": config.weights_path.as_posix()
        if config.weights_path is not None
        else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON artifact."""
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
