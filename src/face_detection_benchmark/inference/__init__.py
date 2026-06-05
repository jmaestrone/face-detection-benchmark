"""Public RF-DETR inference API for face detection workflows."""

from __future__ import annotations

from face_detection_benchmark.inference.images import (
    iter_batches,
    load_coco_image_batch,
    load_frame_image_batch,
    select_device,
    write_preview_image,
)
from face_detection_benchmark.inference.rfdetr_benchmark import (
    DEFAULT_VALIDATION_INFERENCE_THRESHOLD,
    predict_faces_from_coco_dataset,
    rfdetr_model_name_from_weights,
)
from face_detection_benchmark.inference.rfdetr_frames import (
    predict_faces_from_frames,
    read_frame_metadata,
)

load_image_batch = load_frame_image_batch

__all__ = [
    "DEFAULT_VALIDATION_INFERENCE_THRESHOLD",
    "iter_batches",
    "load_coco_image_batch",
    "load_frame_image_batch",
    "load_image_batch",
    "predict_faces_from_coco_dataset",
    "predict_faces_from_frames",
    "read_frame_metadata",
    "rfdetr_model_name_from_weights",
    "select_device",
    "write_preview_image",
]
