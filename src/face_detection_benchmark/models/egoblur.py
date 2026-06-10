"""EgoBlur Gen2 adapter for normalized face benchmark predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from face_detection_benchmark.config import FACE_CATEGORY_NAME
from face_detection_benchmark.predictions import DetectionRecord, xyxy_to_xywh

DEFAULT_EGOBLUR_MODEL_NAME = "egoblur-gen2-face"
DEFAULT_EGOBLUR_FACE_MODEL_PATH = Path("models/egoblur/ego_blur_face_gen2.jit")
DEFAULT_EGOBLUR_CAMERA_NAME = "camera-rgb"
DEFAULT_EGOBLUR_THRESHOLD = 0.005
DEFAULT_EGOBLUR_NMS_IOU_THRESHOLD = 0.5
DEFAULT_EGOBLUR_DEVICE = "auto"
DEFAULT_EGOBLUR_RESIZE_MIN = 1200
DEFAULT_EGOBLUR_RESIZE_MAX = 1200
EGOBLUR_CAMERA_NAMES = (
    "slam-front-left",
    "slam-front-right",
    "slam-side-left",
    "slam-side-right",
    "camera-rgb",
)


@dataclass(frozen=True)
class EgoBlurConfig:
    """Configuration used to load and run an EgoBlur Gen2 face detector."""

    face_model_path: Path
    camera_name: str
    threshold: float
    nms_iou_threshold: float
    device: str
    resize_min: int
    resize_max: int


class EgoBlurDetector:
    """Adapter that converts EgoBlur Gen2 face output into normalized detections."""

    backend = "egoblur"
    generation = "gen2"

    def __init__(self, config: EgoBlurConfig) -> None:
        """Initialize the EgoBlur detector from configuration."""
        self.config = config
        self.detector = self._load_detector(config)

    def predict_batch(self, images_bgr: list[Any]) -> list[list[DetectionRecord]]:
        """Run EgoBlur on BGR images and return normalized face detections."""
        return [self.predict_image(image_bgr) for image_bgr in images_bgr]

    def predict_image(self, image_bgr: Any) -> list[DetectionRecord]:
        """Run EgoBlur on one BGR image and keep face boxes with scores."""
        try:
            import torch
            from gen2.script.detectron2.export.torchscript_patch import (
                patch_instances,
            )
            from gen2.script.predictor import PATCH_INSTANCES_FIELDS
        except ImportError as error:
            raise ValueError(
                "EgoBlur support is not installed. Run "
                "`uv sync --extra egoblur` and try again."
            ) from error

        image_tensor = torch.from_numpy(image_bgr.transpose(2, 0, 1)).to(
            self.detector.device
        )
        with patch_instances(fields=PATCH_INSTANCES_FIELDS):
            detections = self._run_detector_with_scores(image_tensor)
        return egoblur_face_detections_to_records(detections)

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable model configuration metadata."""
        return {
            "generation": self.generation,
            "face_model_path": self.config.face_model_path.as_posix(),
            "face_model_name": self.config.face_model_path.name,
            "camera_name": self.config.camera_name,
            "threshold": self.config.threshold,
            "nms_iou_threshold": self.config.nms_iou_threshold,
            "device": self.config.device,
            "resize_min": self.config.resize_min,
            "resize_max": self.config.resize_max,
        }

    @property
    def last_inference_ms(self) -> float:
        """Return the detector's most recent inference time in milliseconds."""
        seconds = float(getattr(self.detector, "last_inference_time", 0.0))
        return round(seconds * 1000, 4)

    def _load_detector(self, config: EgoBlurConfig) -> Any:
        """Load the configured EgoBlur Gen2 detector implementation."""
        try:
            from gen2.script.predictor import ClassID, EgoblurDetector
        except ImportError as error:
            raise ValueError(
                "EgoBlur support is not installed. Run "
                "`uv sync --extra egoblur` and try again."
            ) from error

        return EgoblurDetector(
            model_path=str(config.face_model_path),
            device=config.device,
            detection_class=ClassID.FACE,
            score_threshold=config.threshold,
            nms_iou_threshold=config.nms_iou_threshold,
            resize_aug={
                "min_size_test": config.resize_min,
                "max_size_test": config.resize_max,
            },
        )

    def _run_detector_with_scores(self, image_tensor: Any) -> Any:
        """Run EgoBlur internals and return FrameDetections with score arrays."""
        batched = image_tensor.unsqueeze(0)
        image_batch, original_sizes, model_input_sizes = self.detector.pre_process(
            batched
        )
        inference_started_at = perf_counter()
        predictions = self.detector.inference(image_batch)
        self.detector.last_inference_time = perf_counter() - inference_started_at
        detections_batch = self.detector.get_detections(
            output_tensor=predictions,
            timestamp_s=0.0,
            stream_id="",
            rotation_angle=0.0,
            model_input_hw_list=model_input_sizes,
            target_img_hw_list=original_sizes,
        )
        if not detections_batch:
            return None
        return detections_batch[0]


def egoblur_face_detections_to_records(detections: Any) -> list[DetectionRecord]:
    """Convert EgoBlur face detections into normalized detection records."""
    if detections is None:
        return []

    boxes = getattr(detections, "face_bboxes", [])
    scores = getattr(detections, "face_scores", [])
    records: list[DetectionRecord] = []
    for index, box in enumerate(boxes):
        bbox_xyxy = [round(float(value), 4) for value in box]
        records.append(
            DetectionRecord(
                bbox_xyxy=bbox_xyxy,
                bbox_xywh=xyxy_to_xywh(bbox_xyxy),
                confidence=round(float(scores[index]), 6),
                class_id=0,
                class_name=FACE_CATEGORY_NAME,
            )
        )
    return records


def resolve_egoblur_device(requested_device: str) -> str:
    """Resolve the EgoBlur device, supporting CUDA or CPU only for v1."""
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        return "cuda:0"
    if requested_device != "auto":
        raise ValueError("--device must be one of: auto, cuda, cpu")

    try:
        import torch
    except ImportError as error:
        raise ValueError(
            "EgoBlur support is not installed. Run "
            "`uv sync --extra egoblur` and try again."
        ) from error

    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"
