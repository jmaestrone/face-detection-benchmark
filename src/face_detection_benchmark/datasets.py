"""Dataset loading helpers for benchmark evaluation inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from face_detection_benchmark.coco import ANNOTATIONS_FILE_NAME


@dataclass(frozen=True)
class CocoImageRecord:
    """One image entry from a COCO detection dataset."""

    id: int
    file_name: str
    width: int
    height: int
    image_path: Path


@dataclass(frozen=True)
class CocoAnnotationRecord:
    """One bounding-box annotation from a COCO detection dataset."""

    id: int
    image_id: int
    category_id: int
    bbox_xywh: list[float]
    area: float
    iscrowd: int


@dataclass(frozen=True)
class CocoDetectionDataset:
    """Loaded COCO detection dataset with image and annotation indexes."""

    root_dir: Path
    annotations_path: Path
    images: list[CocoImageRecord]
    annotations: list[CocoAnnotationRecord]
    categories: list[dict[str, Any]]
    annotations_by_image_id: dict[int, list[CocoAnnotationRecord]] = field(
        default_factory=dict
    )


def load_coco_detection_dataset(
    dataset_dir: Path,
    annotations_path: Path | None = None,
) -> CocoDetectionDataset:
    """Load a COCO detection dataset from a split directory."""
    resolved_annotations_path = annotations_path or dataset_dir / ANNOTATIONS_FILE_NAME
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    if not resolved_annotations_path.exists():
        raise ValueError(
            f"COCO annotations file does not exist: {resolved_annotations_path}"
        )

    payload = json.loads(resolved_annotations_path.read_text(encoding="utf-8"))
    images = [
        CocoImageRecord(
            id=int(row["id"]),
            file_name=str(row["file_name"]),
            width=int(row["width"]),
            height=int(row["height"]),
            image_path=dataset_dir / str(row["file_name"]),
        )
        for row in payload.get("images", [])
    ]
    annotations = [
        CocoAnnotationRecord(
            id=int(row["id"]),
            image_id=int(row["image_id"]),
            category_id=int(row["category_id"]),
            bbox_xywh=[float(value) for value in row["bbox"]],
            area=float(row.get("area", 0.0)),
            iscrowd=int(row.get("iscrowd", 0)),
        )
        for row in payload.get("annotations", [])
    ]
    annotations_by_image_id: dict[int, list[CocoAnnotationRecord]] = {}
    for annotation in annotations:
        annotations_by_image_id.setdefault(annotation.image_id, []).append(annotation)

    return CocoDetectionDataset(
        root_dir=dataset_dir,
        annotations_path=resolved_annotations_path,
        images=images,
        annotations=annotations,
        categories=list(payload.get("categories", [])),
        annotations_by_image_id=annotations_by_image_id,
    )
