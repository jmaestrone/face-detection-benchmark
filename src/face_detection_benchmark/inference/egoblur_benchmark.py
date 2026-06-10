"""EgoBlur prediction workflow for COCO benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
)
from face_detection_benchmark.datasets import load_coco_detection_dataset
from face_detection_benchmark.inference.images import (
    iter_batches,
    load_coco_image_batch,
    write_preview_image,
)
from face_detection_benchmark.models.egoblur import (
    DEFAULT_EGOBLUR_CAMERA_NAME,
    DEFAULT_EGOBLUR_FACE_MODEL_PATH,
    DEFAULT_EGOBLUR_MODEL_NAME,
    DEFAULT_EGOBLUR_NMS_IOU_THRESHOLD,
    DEFAULT_EGOBLUR_RESIZE_MAX,
    DEFAULT_EGOBLUR_RESIZE_MIN,
    DEFAULT_EGOBLUR_THRESHOLD,
    EGOBLUR_CAMERA_NAMES,
    EgoBlurConfig,
    EgoBlurDetector,
    resolve_egoblur_device,
)
from face_detection_benchmark.predictions import (
    ImagePredictionRecord,
    PredictionResult,
    prediction_record_to_json,
    summarize_latency,
    write_latency_summary,
)


def _validate_egoblur_prediction_options(
    dataset_dir: Path,
    face_model_path: Path,
    camera_name: str,
    threshold: float,
    nms_iou_threshold: float,
    resize_min: int,
    resize_max: int,
    batch_size: int,
    device: str,
    limit: int | None,
    max_previews: int,
) -> None:
    """Validate options for benchmark EgoBlur prediction."""
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    if not face_model_path.exists():
        raise ValueError(f"EgoBlur face model file does not exist: {face_model_path}")
    if camera_name not in EGOBLUR_CAMERA_NAMES:
        allowed = ", ".join(EGOBLUR_CAMERA_NAMES)
        raise ValueError(f"--camera-name must be one of: {allowed}")
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if not 0 <= nms_iou_threshold <= 1:
        raise ValueError("--nms-iou-threshold must be between 0 and 1")
    if resize_min <= 0:
        raise ValueError("--resize-min must be greater than 0")
    if resize_max <= 0:
        raise ValueError("--resize-max must be greater than 0")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("--device must be one of: auto, cuda, cpu")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if max_previews < 0:
        raise ValueError("--max-previews must be greater than or equal to 0")


def predict_egoblur_from_coco_dataset(
    dataset_dir: Path = DEFAULT_BENCHMARK_DATA_DIR
    / DEFAULT_BENCHMARK_DATASET_NAME
    / DEFAULT_ROBOFLOW_TEST_SPLIT,
    output_path: Path = DEFAULT_PREDICTIONS_PATH,
    model_name: str = DEFAULT_EGOBLUR_MODEL_NAME,
    face_model_path: Path = DEFAULT_EGOBLUR_FACE_MODEL_PATH,
    camera_name: str = DEFAULT_EGOBLUR_CAMERA_NAME,
    threshold: float = DEFAULT_EGOBLUR_THRESHOLD,
    nms_iou_threshold: float = DEFAULT_EGOBLUR_NMS_IOU_THRESHOLD,
    resize_min: int = DEFAULT_EGOBLUR_RESIZE_MIN,
    resize_max: int = DEFAULT_EGOBLUR_RESIZE_MAX,
    batch_size: int = 4,
    device: str = "auto",
    limit: int | None = None,
    preview_dir: Path | None = None,
    max_previews: int = 20,
    latency_path: Path | None = None,
    latency_table_path: Path | None = None,
) -> PredictionResult:
    """Run EgoBlur Gen2 face detection on every image in a COCO dataset split."""
    _validate_egoblur_prediction_options(
        dataset_dir=dataset_dir,
        face_model_path=face_model_path,
        camera_name=camera_name,
        threshold=threshold,
        nms_iou_threshold=nms_iou_threshold,
        resize_min=resize_min,
        resize_max=resize_max,
        batch_size=batch_size,
        device=device,
        limit=limit,
        max_previews=max_previews,
    )
    dataset = load_coco_detection_dataset(dataset_dir)
    image_records = dataset.images[:limit] if limit is not None else dataset.images
    if not image_records:
        raise ValueError(f"No COCO image records found in {dataset_dir}")

    resolved_device = resolve_egoblur_device(device)
    detector = EgoBlurDetector(
        EgoBlurConfig(
            face_model_path=face_model_path,
            camera_name=camera_name,
            threshold=threshold,
            nms_iou_threshold=nms_iou_threshold,
            device=resolved_device,
            resize_min=resize_min,
            resize_max=resize_max,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    detection_count = 0
    preview_count = 0
    inference_times_ms: list[float] = []
    run_started_at = perf_counter()
    with output_path.open("w", encoding="utf-8") as predictions_file:
        for batch in iter_batches(image_records, batch_size):
            _images_rgb, images_bgr, batch_records = load_coco_image_batch(batch)
            for image_record, image_bgr in zip(batch_records, images_bgr):
                detections = detector.predict_batch([image_bgr])[0]
                inference_ms = detector.last_inference_ms
                prediction_record = ImagePredictionRecord(
                    file_name=image_record.file_name,
                    image_path=image_record.image_path.as_posix(),
                    width=image_record.width,
                    height=image_record.height,
                    detections=detections,
                    model_name=model_name,
                    model_config=detector.metadata(),
                    threshold=threshold,
                    device=resolved_device,
                    backend=detector.backend,
                    timing_ms={"inference": inference_ms},
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
                inference_times_ms.append(inference_ms)

                if preview_dir is not None and preview_count < max_previews:
                    write_preview_image(
                        image_bgr=image_bgr,
                        detections=detections,
                        output_path=preview_dir / image_record.file_name,
                    )
                    preview_count += 1

    if latency_path is not None:
        write_latency_summary(
            summary=summarize_latency(
                model_name=model_name,
                backend=detector.backend,
                device=resolved_device,
                image_count=image_count,
                detection_count=detection_count,
                inference_times_ms=inference_times_ms,
                total_runtime_ms=(perf_counter() - run_started_at) * 1000,
                model_config=detector.metadata(),
            ),
            latency_path=latency_path,
            latency_table_path=latency_table_path,
        )

    return PredictionResult(
        output_path=output_path,
        image_count=image_count,
        detection_count=detection_count,
        preview_dir=preview_dir,
        preview_count=preview_count,
        latency_path=latency_path,
        latency_table_path=latency_table_path,
    )
