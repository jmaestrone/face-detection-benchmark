"""RF-DETR prediction workflow for COCO benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path

from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
)
from face_detection_benchmark.datasets import load_coco_detection_dataset
from face_detection_benchmark.inference.images import (
    iter_batches,
    load_coco_image_batch,
    select_device,
    write_preview_image,
)
from face_detection_benchmark.models.rfdetr import RfdetrConfig, RfdetrFaceDetector
from face_detection_benchmark.predictions import (
    ImagePredictionRecord,
    PredictionResult,
    prediction_record_to_json,
)

DEFAULT_VALIDATION_INFERENCE_THRESHOLD = 0.005


def _validate_benchmark_prediction_options(
    dataset_dir: Path,
    weights_path: Path,
    threshold: float,
    batch_size: int,
    max_detections: int,
    device: str,
    limit: int | None,
    max_previews: int,
) -> None:
    """Validate options for benchmark RF-DETR prediction."""
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    if not weights_path.exists():
        raise ValueError(f"RF-DETR weights file does not exist: {weights_path}")
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if max_detections <= 0:
        raise ValueError("--max-detections must be greater than 0")
    if device not in {"auto", "mps", "cuda", "cpu"}:
        raise ValueError("--device must be one of: auto, mps, cuda, cpu")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if max_previews < 0:
        raise ValueError("--max-previews must be greater than or equal to 0")


def rfdetr_model_name_from_weights(weights_path: Path) -> str:
    """Build a readable RF-DETR model name from a checkpoint path."""
    return f"rfdetr-{weights_path.stem.replace('_', '-')}"


def predict_faces_from_coco_dataset(
    dataset_dir: Path = DEFAULT_BENCHMARK_DATA_DIR
    / DEFAULT_BENCHMARK_DATASET_NAME
    / DEFAULT_ROBOFLOW_TEST_SPLIT,
    output_path: Path = DEFAULT_PREDICTIONS_PATH,
    weights_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_VALIDATION_INFERENCE_THRESHOLD,
    batch_size: int = 4,
    max_detections: int = 40,
    device: str = "auto",
    limit: int | None = None,
    preview_dir: Path | None = None,
    max_previews: int = 20,
    model_name: str | None = None,
) -> PredictionResult:
    """Run RF-DETR on every image in a COCO dataset split."""
    _validate_benchmark_prediction_options(
        dataset_dir=dataset_dir,
        weights_path=weights_path,
        threshold=threshold,
        batch_size=batch_size,
        max_detections=max_detections,
        device=device,
        limit=limit,
        max_previews=max_previews,
    )
    dataset = load_coco_detection_dataset(dataset_dir)
    image_records = dataset.images[:limit] if limit is not None else dataset.images
    if not image_records:
        raise ValueError(f"No COCO image records found in {dataset_dir}")

    selected_device = select_device(device)
    detector = RfdetrFaceDetector(
        RfdetrConfig(
            weights_path=weights_path,
            device=selected_device,
            threshold=threshold,
            max_detections=max_detections,
        )
    )
    resolved_model_name = model_name or rfdetr_model_name_from_weights(weights_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    detection_count = 0
    preview_count = 0
    with output_path.open("w", encoding="utf-8") as predictions_file:
        for batch in iter_batches(image_records, batch_size):
            images_rgb, images_bgr, batch_records = load_coco_image_batch(batch)
            batch_detections = detector.predict_batch(images_rgb)

            for image_record, image_bgr, detections in zip(
                batch_records,
                images_bgr,
                batch_detections,
            ):
                prediction_record = ImagePredictionRecord(
                    file_name=image_record.file_name,
                    image_path=image_record.image_path.as_posix(),
                    width=image_record.width,
                    height=image_record.height,
                    detections=detections,
                    model_name=resolved_model_name,
                    model_config=detector.metadata(),
                    threshold=threshold,
                    device=selected_device,
                    backend=detector.backend,
                )
                predictions_file.write(
                    json.dumps(
                        prediction_record_to_json(prediction_record),
                        sort_keys=True,
                    )
                    + "\n"
                )
                image_count += 1
                detection_count += len(detections)

                if preview_dir is not None and preview_count < max_previews:
                    write_preview_image(
                        image_bgr=image_bgr,
                        detections=detections,
                        output_path=preview_dir / image_record.file_name,
                    )
                    preview_count += 1

    return PredictionResult(
        output_path=output_path,
        image_count=image_count,
        detection_count=detection_count,
        preview_dir=preview_dir,
        preview_count=preview_count,
    )
