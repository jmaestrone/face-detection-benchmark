"""RF-DETR model adapter for normalized face detection predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from face_detection_benchmark.config import FACE_CATEGORY_NAME
from face_detection_benchmark.predictions import DetectionRecord, xyxy_to_xywh


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
