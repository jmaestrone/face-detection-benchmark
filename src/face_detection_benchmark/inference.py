"""RF-DETR inference helpers for extracted face frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from face_detection_benchmark.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAMES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    FACE_CATEGORY_NAME,
)
from face_detection_benchmark.models.rfdetr import RfdetrConfig, RfdetrFaceDetector
from face_detection_benchmark.predictions import (
    DetectionRecord,
    ImagePredictionRecord,
    PredictionResult,
    prediction_record_to_json,
)
from face_detection_benchmark.video import METADATA_FILE_NAME, FrameMetadata


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
    _validate_prediction_options(
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
            images_rgb, images_bgr, batch_records = load_image_batch(frames_dir, batch)
            batch_detections = detector.predict_batch(images_rgb)

            for frame_record, image_bgr, detections in zip(
                batch_records, images_bgr, batch_detections
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


def read_frame_metadata(
    metadata_path: Path,
    limit: int | None = None,
) -> list[FrameMetadata]:
    """Read extraction metadata written by the frame extraction tool."""
    records: list[FrameMetadata] = []
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(FrameMetadata(**payload))
            if limit is not None and len(records) >= limit:
                break
    return records


def iter_batches(
    records: list[FrameMetadata],
    batch_size: int,
) -> Iterable[list[FrameMetadata]]:
    """Yield frame metadata in fixed-size batches."""
    for start_index in range(0, len(records), batch_size):
        yield records[start_index : start_index + batch_size]


def load_image_batch(
    frames_dir: Path,
    records: list[FrameMetadata],
) -> tuple[list[Any], list[Any], list[FrameMetadata]]:
    """Load images as RGB for RF-DETR and BGR for optional previews."""
    import cv2

    images_rgb = []
    images_bgr = []
    loaded_records = []
    for record in records:
        image_path = frames_dir / record.output_path
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Could not read extracted frame: {image_path}")
        images_bgr.append(image_bgr)
        images_rgb.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        loaded_records.append(record)

    return images_rgb, images_bgr, loaded_records


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


def write_preview_image(
    image_bgr: Any,
    detections: list[DetectionRecord],
    output_path: Path,
) -> None:
    """Write an annotated preview image with detection boxes and confidences."""
    import cv2

    annotated = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = [round(value) for value in detection.bbox_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = FACE_CATEGORY_NAME
        if detection.confidence is not None:
            label = f"{label} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), annotated)


def _validate_prediction_options(
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
