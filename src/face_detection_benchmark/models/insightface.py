"""InsightFace/SCRFD model adapter for normalized benchmark predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from face_detection_benchmark.config import FACE_CATEGORY_NAME
from face_detection_benchmark.predictions import DetectionRecord, xyxy_to_xywh

DEFAULT_INSIGHTFACE_MODEL_NAME = "insightface-buffalo-l"
DEFAULT_INSIGHTFACE_MODEL_PACK = "buffalo_l"
DEFAULT_INSIGHTFACE_DET_SIZE = 960
DEFAULT_INSIGHTFACE_THRESHOLD = 0.005
DEFAULT_INSIGHTFACE_PROVIDERS = ("CPUExecutionProvider",)
DEFAULT_INSIGHTFACE_CTX_ID = -1


@dataclass(frozen=True)
class InsightFaceConfig:
    """Configuration used to load and run an InsightFace detector."""

    model_pack: str
    providers: tuple[str, ...]
    ctx_id: int
    det_size: int
    threshold: float


class InsightFaceDetector:
    """Adapter that converts InsightFace detector output into normalized detections."""

    backend = "insightface"

    def __init__(self, config: InsightFaceConfig) -> None:
        """Initialize the InsightFace detector from configuration."""
        self.config = config
        self.app = self._load_app(config)
        self.detector = self.app.models.get("detection")
        if self.detector is None:
            raise ValueError("InsightFace model pack did not load a detection model")

    def predict_batch(self, images_bgr: list[Any]) -> list[list[DetectionRecord]]:
        """Run InsightFace/SCRFD on BGR images and return normalized detections."""
        return [
            detector_output_to_records(self.detector.detect(image_bgr))
            for image_bgr in images_bgr
        ]

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable model configuration metadata."""
        return {
            "model_pack": self.config.model_pack,
            "providers": list(self.config.providers),
            "ctx_id": self.config.ctx_id,
            "det_size": self.config.det_size,
            "threshold": self.config.threshold,
        }

    def _load_app(self, config: InsightFaceConfig) -> Any:
        """Load and prepare the configured InsightFace application."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise ValueError(
                "InsightFace support is not installed. Run "
                "`uv sync --extra insightface` and try again."
            ) from error

        app = FaceAnalysis(
            name=config.model_pack,
            allowed_modules=["detection"],
            providers=list(config.providers),
        )
        app.prepare(
            ctx_id=config.ctx_id,
            det_size=(config.det_size, config.det_size),
            det_thresh=config.threshold,
        )
        return app


def parse_providers(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse provider names from a comma-separated string or iterable."""
    if isinstance(value, str):
        providers = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        providers = tuple(
            provider_name
            for provider in value
            if (provider_name := str(provider).strip())
        )
    if not providers:
        raise ValueError("--providers must include at least one provider")
    return providers


def detector_output_to_records(output: Any) -> list[DetectionRecord]:
    """Convert raw InsightFace detector output into normalized detection records."""
    detections = output[0] if isinstance(output, tuple) else output
    return detections_to_records(detections)


def detections_to_records(detections: Any) -> list[DetectionRecord]:
    """Convert InsightFace detector boxes into normalized detection records."""
    records: list[DetectionRecord] = []
    if detections is None:
        return records
    for detection in detections:
        values = [float(value) for value in detection]
        if len(values) < 5:
            continue
        bbox_xyxy = [round(value, 4) for value in values[:4]]
        records.append(
            DetectionRecord(
                bbox_xyxy=bbox_xyxy,
                bbox_xywh=xyxy_to_xywh(bbox_xyxy),
                confidence=round(values[4], 6),
                class_id=None,
                class_name=FACE_CATEGORY_NAME,
            )
        )
    return records
