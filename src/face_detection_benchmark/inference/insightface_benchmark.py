"""InsightFace prediction workflow for COCO benchmark datasets."""

from __future__ import annotations

import json
from pathlib import Path

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
from face_detection_benchmark.models.insightface import (
    DEFAULT_INSIGHTFACE_CTX_ID,
    DEFAULT_INSIGHTFACE_DET_SIZE,
    DEFAULT_INSIGHTFACE_MODEL_NAME,
    DEFAULT_INSIGHTFACE_MODEL_PACK,
    DEFAULT_INSIGHTFACE_PROVIDERS,
    DEFAULT_INSIGHTFACE_THRESHOLD,
    InsightFaceConfig,
    InsightFaceDetector,
)
from face_detection_benchmark.predictions import (
    ImagePredictionRecord,
    PredictionResult,
    prediction_record_to_json,
)


def _validate_insightface_prediction_options(
    dataset_dir: Path,
    threshold: float,
    batch_size: int,
    det_size: int,
    limit: int | None,
    max_previews: int,
) -> None:
    """Validate options for benchmark InsightFace prediction."""
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if det_size <= 0:
        raise ValueError("--det-size must be greater than 0")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if max_previews < 0:
        raise ValueError("--max-previews must be greater than or equal to 0")


def predict_insightface_from_coco_dataset(
    dataset_dir: Path = DEFAULT_BENCHMARK_DATA_DIR
    / DEFAULT_BENCHMARK_DATASET_NAME
    / DEFAULT_ROBOFLOW_TEST_SPLIT,
    output_path: Path = DEFAULT_PREDICTIONS_PATH,
    model_name: str = DEFAULT_INSIGHTFACE_MODEL_NAME,
    model_pack: str = DEFAULT_INSIGHTFACE_MODEL_PACK,
    threshold: float = DEFAULT_INSIGHTFACE_THRESHOLD,
    det_size: int = DEFAULT_INSIGHTFACE_DET_SIZE,
    providers: tuple[str, ...] = DEFAULT_INSIGHTFACE_PROVIDERS,
    ctx_id: int = DEFAULT_INSIGHTFACE_CTX_ID,
    batch_size: int = 4,
    limit: int | None = None,
    preview_dir: Path | None = None,
    max_previews: int = 20,
) -> PredictionResult:
    """Run InsightFace/SCRFD on every image in a COCO dataset split."""
    _validate_insightface_prediction_options(
        dataset_dir=dataset_dir,
        threshold=threshold,
        batch_size=batch_size,
        det_size=det_size,
        limit=limit,
        max_previews=max_previews,
    )
    dataset = load_coco_detection_dataset(dataset_dir)
    image_records = dataset.images[:limit] if limit is not None else dataset.images
    if not image_records:
        raise ValueError(f"No COCO image records found in {dataset_dir}")

    detector = InsightFaceDetector(
        InsightFaceConfig(
            model_pack=model_pack,
            providers=providers,
            ctx_id=ctx_id,
            det_size=det_size,
            threshold=threshold,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    detection_count = 0
    preview_count = 0
    with output_path.open("w", encoding="utf-8") as predictions_file:
        for batch in iter_batches(image_records, batch_size):
            _images_rgb, images_bgr, batch_records = load_coco_image_batch(batch)
            batch_detections = detector.predict_batch(images_bgr)

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
                    model_name=model_name,
                    model_config=detector.metadata(),
                    threshold=threshold,
                    device="cpu" if ctx_id < 0 else f"ctx:{ctx_id}",
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
