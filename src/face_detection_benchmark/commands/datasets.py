"""Dataset export, download, and upload CLI command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from face_detection_benchmark.coco import export_predictions_to_coco
from face_detection_benchmark.commands.common import optional_int_env
from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_BENCHMARK_IMAGE_COUNT,
    DEFAULT_FRAMES_DIR,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_ROBOFLOW_EXPORT_DIR,
    DEFAULT_ROBOFLOW_FORMAT,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
    FACE_CATEGORY_NAME,
)
from face_detection_benchmark.datasets import download_roboflow_benchmark_dataset
from face_detection_benchmark.env import get_env_value


def export_coco(
    frames_dir: Annotated[
        Path,
        typer.Option(
            "--frames-dir",
            help="Directory containing extracted frames and metadata.jsonl.",
        ),
    ] = DEFAULT_FRAMES_DIR,
    predictions_path: Annotated[
        Path,
        typer.Option(
            "--predictions-path",
            help="JSONL predictions path written by predict-faces.",
        ),
    ] = DEFAULT_PREDICTIONS_PATH,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where the Roboflow-ready COCO export is written.",
        ),
    ] = DEFAULT_ROBOFLOW_EXPORT_DIR,
    include_empty: Annotated[
        bool,
        typer.Option(
            "--include-empty",
            help="Include frames with no detections in the exported dataset.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite image files already present in the export directory.",
        ),
    ] = False,
) -> None:
    """Export RF-DETR detections as COCO ground-truth annotations."""
    try:
        result = export_predictions_to_coco(
            frames_dir=frames_dir,
            predictions_path=predictions_path,
            output_dir=output_dir,
            include_empty=include_empty,
            overwrite=overwrite,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Exported {result.image_count} images and {result.annotation_count} "
        f"annotations to {result.dataset_dir}"
    )
    typer.echo(f"COCO annotations: {result.annotations_path}")
    if result.clipped_box_count:
        typer.echo(f"Clipped boxes at image bounds: {result.clipped_box_count}")
    if result.skipped_box_count:
        typer.echo(f"Skipped invalid boxes: {result.skipped_box_count}")


def download_roboflow_benchmark(
    workspace: Annotated[
        str | None,
        typer.Option(
            "--workspace",
            help="Roboflow workspace slug.",
        ),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option(
            "--project",
            help="Roboflow project slug.",
        ),
    ] = None,
    version: Annotated[
        int | None,
        typer.Option(
            "--version",
            min=1,
            help="Roboflow dataset version number.",
        ),
    ] = None,
    dataset_name: Annotated[
        str,
        typer.Option(
            "--dataset-name",
            help="Local benchmark dataset directory name.",
        ),
    ] = DEFAULT_BENCHMARK_DATASET_NAME,
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            help="Root directory for ignored benchmark datasets.",
        ),
    ] = DEFAULT_BENCHMARK_DATA_DIR,
    model_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Roboflow export format identifier.",
        ),
    ] = DEFAULT_ROBOFLOW_FORMAT,
    expected_split: Annotated[
        str,
        typer.Option(
            "--expected-split",
            help="Required benchmark split name.",
        ),
    ] = DEFAULT_ROBOFLOW_TEST_SPLIT,
    expected_category: Annotated[
        str,
        typer.Option(
            "--expected-category",
            help="Required face category name in the COCO export.",
        ),
    ] = FACE_CATEGORY_NAME,
    expected_image_count: Annotated[
        int,
        typer.Option(
            "--expected-image-count",
            min=0,
            help="Required image count, or 0 to skip the count check.",
        ),
    ] = DEFAULT_BENCHMARK_IMAGE_COUNT,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite an existing local Roboflow dataset download.",
        ),
    ] = False,
) -> None:
    """Download and validate the cleaned Roboflow test benchmark dataset."""
    try:
        resolved_workspace = workspace or get_env_value("ROBOFLOW_WORKSPACE")
        resolved_project = project or get_env_value("ROBOFLOW_PROJECT")
        resolved_version = version or optional_int_env("ROBOFLOW_VERSION")
        if resolved_workspace is None:
            raise ValueError("--workspace or ROBOFLOW_WORKSPACE is required")
        if resolved_project is None:
            raise ValueError("--project or ROBOFLOW_PROJECT is required")
        if resolved_version is None:
            raise ValueError("--version or ROBOFLOW_VERSION is required")
        result = download_roboflow_benchmark_dataset(
            workspace=resolved_workspace,
            project=resolved_project,
            version=resolved_version,
            dataset_name=dataset_name,
            output_root=output_root,
            model_format=model_format,
            expected_split=expected_split,
            expected_category=expected_category,
            expected_image_count=expected_image_count or None,
            overwrite=overwrite,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Downloaded Roboflow benchmark {result.workspace}/{result.project}/"
        f"{result.version} to {result.dataset_dir}"
    )
    typer.echo(
        f"Validated {result.image_count} images and {result.annotation_count} "
        f"annotations in {result.split_dir}"
    )
    typer.echo(f"COCO annotations: {result.annotations_path}")


def upload_roboflow() -> None:
    """Upload the exported COCO dataset to Roboflow."""
    typer.echo("upload-roboflow is planned for a later checkpoint.")
