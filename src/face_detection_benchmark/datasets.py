"""Dataset loading helpers for benchmark evaluation inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from face_detection_benchmark.coco import ANNOTATIONS_FILE_NAME
from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_BENCHMARK_IMAGE_COUNT,
    DEFAULT_ROBOFLOW_FORMAT,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
    FACE_CATEGORY_NAME,
)
from face_detection_benchmark.env import get_env_value

SPLIT_NAMES = ("train", "valid", "test")


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


@dataclass(frozen=True)
class RoboflowBenchmarkDownloadResult:
    """Summary of a Roboflow benchmark dataset download."""

    dataset_dir: Path
    split_dir: Path
    annotations_path: Path
    workspace: str
    project: str
    version: int
    model_format: str
    image_count: int
    annotation_count: int
    categories: list[str]


def download_roboflow_benchmark_dataset(
    workspace: str,
    project: str,
    version: int,
    dataset_name: str = DEFAULT_BENCHMARK_DATASET_NAME,
    output_root: Path = DEFAULT_BENCHMARK_DATA_DIR,
    model_format: str = DEFAULT_ROBOFLOW_FORMAT,
    expected_split: str = DEFAULT_ROBOFLOW_TEST_SPLIT,
    expected_category: str = FACE_CATEGORY_NAME,
    expected_image_count: int | None = DEFAULT_BENCHMARK_IMAGE_COUNT,
    overwrite: bool = False,
    api_key: str | None = None,
) -> RoboflowBenchmarkDownloadResult:
    """Download and validate the cleaned Roboflow benchmark dataset."""
    resolved_api_key = api_key or get_env_value("ROBOFLOW_API_KEY")
    if not resolved_api_key:
        raise ValueError("ROBOFLOW_API_KEY is not set in the environment or .env")
    if expected_split not in SPLIT_NAMES:
        raise ValueError(f"Expected split must be one of: {', '.join(SPLIT_NAMES)}")
    if version <= 0:
        raise ValueError("--version must be greater than 0")

    dataset_dir = output_root / dataset_name
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    from roboflow import Roboflow

    roboflow = Roboflow(api_key=resolved_api_key)
    roboflow.workspace(workspace).project(project).version(version).download(
        model_format=model_format,
        location=str(dataset_dir),
        overwrite=overwrite,
    )

    result = validate_roboflow_benchmark_dataset(
        dataset_dir=dataset_dir,
        workspace=workspace,
        project=project,
        version=version,
        model_format=model_format,
        expected_split=expected_split,
        expected_category=expected_category,
        expected_image_count=expected_image_count,
    )
    write_roboflow_source_metadata(result)
    return result


def validate_roboflow_benchmark_dataset(
    dataset_dir: Path,
    workspace: str = "",
    project: str = "",
    version: int = 0,
    model_format: str = DEFAULT_ROBOFLOW_FORMAT,
    expected_split: str = DEFAULT_ROBOFLOW_TEST_SPLIT,
    expected_category: str = FACE_CATEGORY_NAME,
    expected_image_count: int | None = DEFAULT_BENCHMARK_IMAGE_COUNT,
) -> RoboflowBenchmarkDownloadResult:
    """Validate that a Roboflow COCO export is a test-only benchmark dataset."""
    if expected_split not in SPLIT_NAMES:
        raise ValueError(f"Expected split must be one of: {', '.join(SPLIT_NAMES)}")

    split_dir = dataset_dir / expected_split
    annotations_path = split_dir / ANNOTATIONS_FILE_NAME
    dataset = load_coco_detection_dataset(split_dir, annotations_path)

    category_names = [str(category.get("name")) for category in dataset.categories]
    if expected_category not in category_names:
        raise ValueError(
            f"Expected category {expected_category!r} in {annotations_path}; "
            f"found {category_names}"
        )
    if not dataset.images:
        raise ValueError(f"No images found in benchmark split: {split_dir}")
    missing_image_paths = [
        image.image_path for image in dataset.images if not image.image_path.exists()
    ]
    if missing_image_paths:
        preview_paths = ", ".join(path.as_posix() for path in missing_image_paths[:3])
        raise ValueError(
            f"COCO annotations reference {len(missing_image_paths)} missing images; "
            f"first missing paths: {preview_paths}"
        )
    if expected_image_count is not None and len(dataset.images) != expected_image_count:
        raise ValueError(
            f"Expected {expected_image_count} images in {split_dir}; "
            f"found {len(dataset.images)}"
        )

    for split_name in SPLIT_NAMES:
        if split_name == expected_split:
            continue
        split_annotations_path = dataset_dir / split_name / ANNOTATIONS_FILE_NAME
        if not split_annotations_path.exists():
            continue
        split_payload = json.loads(split_annotations_path.read_text(encoding="utf-8"))
        image_count = len(split_payload.get("images", []))
        annotation_count = len(split_payload.get("annotations", []))
        if image_count or annotation_count:
            raise ValueError(
                f"Benchmark dataset must be test-only; found {image_count} images "
                f"and {annotation_count} annotations in {split_annotations_path}"
            )

    return RoboflowBenchmarkDownloadResult(
        dataset_dir=dataset_dir,
        split_dir=split_dir,
        annotations_path=annotations_path,
        workspace=workspace,
        project=project,
        version=version,
        model_format=model_format,
        image_count=len(dataset.images),
        annotation_count=len(dataset.annotations),
        categories=category_names,
    )


def write_roboflow_source_metadata(
    result: RoboflowBenchmarkDownloadResult,
) -> Path:
    """Write non-secret source metadata for a downloaded Roboflow benchmark."""
    metadata_path = result.dataset_dir / "roboflow_source.json"
    payload = {
        "workspace": result.workspace,
        "project": result.project,
        "version": result.version,
        "format": result.model_format,
        "split": result.split_dir.name,
        "annotations_path": result.annotations_path.as_posix(),
        "image_count": result.image_count,
        "annotation_count": result.annotation_count,
        "categories": result.categories,
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata_path


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
