"""COCO export helpers for Roboflow dataset uploads."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from face_detection_benchmark.config import (
    DEFAULT_FRAMES_DIR,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_ROBOFLOW_EXPORT_DIR,
    FACE_CATEGORY_NAME,
)

ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
TRAIN_SPLIT_NAME = "train"


@dataclass(frozen=True)
class CocoExportResult:
    """Summary of a COCO export run."""

    dataset_dir: Path
    annotations_path: Path
    image_count: int
    annotation_count: int
    clipped_box_count: int
    skipped_box_count: int


def export_predictions_to_coco(
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    predictions_path: Path = DEFAULT_PREDICTIONS_PATH,
    output_dir: Path = DEFAULT_ROBOFLOW_EXPORT_DIR,
    include_empty: bool = False,
    overwrite: bool = False,
) -> CocoExportResult:
    """Export RF-DETR prediction JSONL as a COCO ground-truth dataset."""
    _validate_export_inputs(frames_dir, predictions_path)

    prediction_rows = read_prediction_rows(predictions_path)
    if not prediction_rows:
        raise ValueError(f"No prediction rows found in {predictions_path}")

    dataset_dir = output_dir / TRAIN_SPLIT_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = dataset_dir / ANNOTATIONS_FILE_NAME

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    clipped_box_count = 0
    skipped_box_count = 0
    next_image_id = 1
    next_annotation_id = 1

    for row in prediction_rows:
        detections = list(row.get("detections", []))
        if not detections and not include_empty:
            continue

        source_image_path = frames_dir / row["file_name"]
        if not source_image_path.exists():
            raise ValueError(f"Frame image does not exist: {source_image_path}")

        exported_file_name = row["file_name"]
        destination_image_path = dataset_dir / exported_file_name
        copy_image_file(source_image_path, destination_image_path, overwrite=overwrite)

        image_id = next_image_id
        next_image_id += 1
        width = int(row["width"])
        height = int(row["height"])
        images.append(
            {
                "id": image_id,
                "file_name": exported_file_name,
                "width": width,
                "height": height,
            }
        )

        for detection in detections:
            clipped_bbox, was_clipped = clip_xyxy_to_image(
                detection["bbox_xyxy"],
                width=width,
                height=height,
            )
            if was_clipped:
                clipped_box_count += 1

            bbox_xywh = xyxy_to_xywh(clipped_bbox)
            if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
                skipped_box_count += 1
                continue

            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox_xywh,
                    "area": round(bbox_xywh[2] * bbox_xywh[3], 4),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            next_annotation_id += 1

    coco_payload = {
        "info": {
            "description": "RF-DETR face detections exported as ground-truth labels",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {
                "id": 1,
                "name": FACE_CATEGORY_NAME,
                "supercategory": "face",
            }
        ],
        "images": images,
        "annotations": annotations,
    }
    annotations_path.write_text(
        json.dumps(coco_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return CocoExportResult(
        dataset_dir=dataset_dir,
        annotations_path=annotations_path,
        image_count=len(images),
        annotation_count=len(annotations),
        clipped_box_count=clipped_box_count,
        skipped_box_count=skipped_box_count,
    )


def read_prediction_rows(predictions_path: Path) -> list[dict[str, Any]]:
    """Read JSONL prediction rows written by the inference command."""
    rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8") as predictions_file:
        for line in predictions_file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def copy_image_file(
    source_image_path: Path,
    destination_image_path: Path,
    overwrite: bool,
) -> None:
    """Copy a frame image into the COCO dataset directory."""
    if destination_image_path.exists() and not overwrite:
        return
    shutil.copy2(source_image_path, destination_image_path)


def clip_xyxy_to_image(
    bbox_xyxy: list[float],
    width: int,
    height: int,
) -> tuple[list[float], bool]:
    """Clip an xyxy box to image bounds and report whether it changed."""
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    clipped = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    rounded = [round(value, 4) for value in clipped]
    original = [round(value, 4) for value in [x1, y1, x2, y2]]
    return rounded, rounded != original


def xyxy_to_xywh(bbox_xyxy: list[float]) -> list[float]:
    """Convert an xyxy box to COCO xywh format."""
    x1, y1, x2, y2 = bbox_xyxy
    return [
        round(x1, 4),
        round(y1, 4),
        round(x2 - x1, 4),
        round(y2 - y1, 4),
    ]


def _validate_export_inputs(frames_dir: Path, predictions_path: Path) -> None:
    """Validate source frame and prediction paths before COCO export."""
    if not frames_dir.exists():
        raise ValueError(f"Frames directory does not exist: {frames_dir}")
    if not predictions_path.exists():
        raise ValueError(f"Predictions file does not exist: {predictions_path}")
