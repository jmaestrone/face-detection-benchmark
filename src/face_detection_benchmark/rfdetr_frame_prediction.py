"""RF-DETR prediction workflow for extracted video frames."""

from __future__ import annotations

import json
from pathlib import Path

from face_detection_benchmark.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAMES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
)
from face_detection_benchmark.inference_images import (
    iter_batches,
    load_frame_image_batch,
    select_device,
    write_preview_image,
)
from face_detection_benchmark.models.rfdetr import RfdetrConfig, RfdetrFaceDetector
from face_detection_benchmark.predictions import (
    ImagePredictionRecord,
    PredictionResult,
    prediction_record_to_json,
)
from face_detection_benchmark.video import METADATA_FILE_NAME, FrameMetadata


def _validate_frame_prediction_options(
    frames_dir: Path,
    metadata_path: Path,
    weights_path: Path,
    threshold: float,
    batch_size: int,
    max_detections: int,
    device: str,
    limit: int | None,
    max_previews: int,
) -> None:
    """Validate options for extracted-frame RF-DETR prediction."""
    if not frames_dir.exists():
        raise ValueError(f"Frames directory does not exist: {frames_dir}")
    if not metadata_path.exists():
        raise ValueError(f"Frame metadata file does not exist: {metadata_path}")
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


def read_frame_metadata(
    metadata_path: Path,
    limit: int | None = None,
) -> list[FrameMetadata]:
    """Read extraction metadata written by the frame extraction tool."""
    frame_records: list[FrameMetadata] = []
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            frame_records.append(FrameMetadata(**payload))
            if limit is not None and len(frame_records) >= limit:
                break
    return frame_records


def predict_faces_from_frames(
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    metadata_path: Path | None = None,
    output_path: Path = DEFAULT_PREDICTIONS_PATH,
    weights_path: Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size: int = 4,
    max_detections: int = 40,
    device: str = "auto",
    limit: int | None = None,
    preview_dir: Path | None = None,
    max_previews: int = 20,
) -> PredictionResult:
    """Run RF-DETR on extracted frame images and write JSONL detections."""
    resolved_metadata_path = metadata_path or frames_dir / METADATA_FILE_NAME
    _validate_frame_prediction_options(
        frames_dir=frames_dir,
        metadata_path=resolved_metadata_path,
        weights_path=weights_path,
        threshold=threshold,
        batch_size=batch_size,
        max_detections=max_detections,
        device=device,
        limit=limit,
        max_previews=max_previews,
    )

    frame_records = read_frame_metadata(resolved_metadata_path, limit=limit)
    if not frame_records:
        raise ValueError(f"No frame records found in {resolved_metadata_path}")

    selected_device = select_device(device)
    detector = RfdetrFaceDetector(
        RfdetrConfig(
            weights_path=weights_path,
            device=selected_device,
            threshold=threshold,
            max_detections=max_detections,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    detection_count = 0
    preview_count = 0

    with output_path.open("w", encoding="utf-8") as predictions_file:
        for batch in iter_batches(frame_records, batch_size):
            images_rgb, images_bgr, batch_records = load_frame_image_batch(
                frames_dir,
                batch,
            )
            batch_detections = detector.predict_batch(images_rgb)

            for frame_record, image_bgr, detections in zip(
                batch_records,
                images_bgr,
                batch_detections,
            ):
                image_record = ImagePredictionRecord(
                    file_name=frame_record.file_name,
                    image_path=(frames_dir / frame_record.output_path).as_posix(),
                    width=frame_record.width,
                    height=frame_record.height,
                    detections=detections,
                    model_name=detector.model_name,
                    model_config=detector.metadata(),
                    source_video=frame_record.source_video,
                    frame_index=frame_record.frame_index,
                    timestamp_seconds=frame_record.timestamp_seconds,
                    threshold=threshold,
                    device=selected_device,
                    backend=detector.backend,
                )
                predictions_file.write(
                    json.dumps(prediction_record_to_json(image_record), sort_keys=True)
                    + "\n"
                )
                image_count += 1
                detection_count += len(detections)

                if preview_dir is not None and preview_count < max_previews:
                    write_preview_image(
                        image_bgr=image_bgr,
                        detections=detections,
                        output_path=preview_dir / frame_record.file_name,
                    )
                    preview_count += 1

    return PredictionResult(
        output_path=output_path,
        image_count=image_count,
        detection_count=detection_count,
        preview_dir=preview_dir,
        preview_count=preview_count,
    )
