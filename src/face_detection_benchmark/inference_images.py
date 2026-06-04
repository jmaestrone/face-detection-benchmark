"""Shared image loading and preview helpers for RF-DETR inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, TypeVar

from face_detection_benchmark.config import FACE_CATEGORY_NAME
from face_detection_benchmark.datasets import CocoImageRecord
from face_detection_benchmark.predictions import DetectionRecord
from face_detection_benchmark.video import FrameMetadata

RecordT = TypeVar("RecordT")


def iter_batches(records: list[RecordT], batch_size: int) -> Iterable[list[RecordT]]:
    """Yield records in fixed-size batches."""
    for start_index in range(0, len(records), batch_size):
        yield records[start_index : start_index + batch_size]


def select_device(requested_device: str) -> str:
    """Choose the RF-DETR device, preferring MPS on Apple Silicon when available."""
    if requested_device != "auto":
        if requested_device not in {"mps", "cuda", "cpu"}:
            raise ValueError("--device must be one of: auto, mps, cuda, cpu")
        return requested_device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_frame_image_batch(
    frames_dir: Path,
    records: list[FrameMetadata],
) -> tuple[list[Any], list[Any], list[FrameMetadata]]:
    """Load extracted frames as RGB for RF-DETR and BGR for previews."""
    import cv2

    images_rgb = []
    images_bgr = []
    loaded_records = []
    for frame_record in records:
        image_path = frames_dir / frame_record.output_path
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Could not read extracted frame: {image_path}")
        images_bgr.append(image_bgr)
        images_rgb.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        loaded_records.append(frame_record)

    return images_rgb, images_bgr, loaded_records


def load_coco_image_batch(
    records: list[CocoImageRecord],
) -> tuple[list[Any], list[Any], list[CocoImageRecord]]:
    """Load COCO dataset images as RGB for RF-DETR and BGR for previews."""
    import cv2

    images_rgb = []
    images_bgr = []
    loaded_records = []
    for image_record in records:
        image_bgr = cv2.imread(str(image_record.image_path))
        if image_bgr is None:
            raise ValueError(
                f"Could not read benchmark image: {image_record.image_path}"
            )
        images_bgr.append(image_bgr)
        images_rgb.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        loaded_records.append(image_record)

    return images_rgb, images_bgr, loaded_records


def write_preview_image(
    image_bgr: Any,
    detections: list[DetectionRecord],
    output_path: Path,
) -> None:
    """Write an annotated preview image with detection boxes and confidences."""
    import cv2

    annotated_image = image_bgr.copy()
    for detection in detections:
        x_min, y_min, x_max, y_max = [round(value) for value in detection.bbox_xyxy]
        cv2.rectangle(
            annotated_image,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2,
        )
        label = FACE_CATEGORY_NAME
        if detection.confidence is not None:
            label = f"{label} {detection.confidence:.2f}"
        cv2.putText(
            annotated_image,
            label,
            (x_min, max(16, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), annotated_image)
