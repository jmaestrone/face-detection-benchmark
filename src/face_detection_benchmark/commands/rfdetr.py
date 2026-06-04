"""RF-DETR prediction CLI command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from face_detection_benchmark.commands.common import default_run_id
from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAMES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
    DEFAULT_RUNS_DIR,
)
from face_detection_benchmark.inference import (
    DEFAULT_VALIDATION_INFERENCE_THRESHOLD,
    predict_faces_from_coco_dataset,
    predict_faces_from_frames,
    rfdetr_model_name_from_weights,
)


def predict_faces(
    frames_dir: Annotated[
        Path,
        typer.Option(
            "--frames-dir",
            help="Directory containing extracted frames and metadata.jsonl.",
        ),
    ] = DEFAULT_FRAMES_DIR,
    metadata_path: Annotated[
        Path | None,
        typer.Option(
            "--metadata-path",
            help="Frame metadata JSONL path. Defaults to <frames-dir>/metadata.jsonl.",
        ),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            "-o",
            help="JSONL path where RF-DETR detections are written.",
        ),
    ] = DEFAULT_PREDICTIONS_PATH,
    weights: Annotated[
        Path,
        typer.Option(
            "--weights",
            help="Local RF-DETR checkpoint path.",
        ),
    ] = DEFAULT_MODEL_PATH,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            max=1.0,
            help="Confidence threshold for RF-DETR detections.",
        ),
    ] = DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Number of frames to predict per RF-DETR batch.",
        ),
    ] = 4,
    max_detections: Annotated[
        int,
        typer.Option(
            "--max-detections",
            min=1,
            help="Maximum face detections requested from RF-DETR per frame.",
        ),
    ] = 40,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device to use: auto, mps, cuda, or cpu.",
        ),
    ] = "auto",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Only process the first N frames for a smoke run.",
        ),
    ] = None,
    preview_dir: Annotated[
        Path | None,
        typer.Option(
            "--preview-dir",
            help="Optional directory for annotated preview images.",
        ),
    ] = None,
    max_previews: Annotated[
        int,
        typer.Option(
            "--max-previews",
            min=0,
            help="Maximum number of preview images to write when --preview-dir is set.",
        ),
    ] = 20,
) -> None:
    """Run RF-DETR face detection on extracted frames."""
    try:
        result = predict_faces_from_frames(
            frames_dir=frames_dir,
            metadata_path=metadata_path,
            output_path=output_path,
            weights_path=weights,
            threshold=threshold,
            batch_size=batch_size,
            max_detections=max_detections,
            device=device,
            limit=limit,
            preview_dir=preview_dir,
            max_previews=max_previews,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Wrote detections for {result.image_count} frames "
        f"({result.detection_count} boxes) to {result.output_path}"
    )
    if result.preview_count:
        typer.echo(f"Preview images: {result.preview_count} in {result.preview_dir}")


def predict_rfdetr_benchmark(
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            help="COCO benchmark split directory containing _annotations.coco.json.",
        ),
    ] = DEFAULT_BENCHMARK_DATA_DIR
    / DEFAULT_BENCHMARK_DATASET_NAME
    / DEFAULT_ROBOFLOW_TEST_SPLIT,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output-path",
            "-o",
            help=(
                "JSONL output path. Defaults to "
                "runs/benchmarks/<run-id>/predictions/<model>.jsonl."
            ),
        ),
    ] = None,
    weights: Annotated[
        Path,
        typer.Option(
            "--weights",
            help="Local RF-DETR checkpoint path.",
        ),
    ] = DEFAULT_MODEL_PATH,
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model-name",
            help="Model name written into prediction rows.",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Run id used when --output-path is not supplied.",
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            max=1.0,
            help="Low RF-DETR inference threshold used before validation sweeps.",
        ),
    ] = DEFAULT_VALIDATION_INFERENCE_THRESHOLD,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Number of images to predict per RF-DETR batch.",
        ),
    ] = 4,
    max_detections: Annotated[
        int,
        typer.Option(
            "--max-detections",
            min=1,
            help="Maximum face detections requested from RF-DETR per image.",
        ),
    ] = 40,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device to use: auto, mps, cuda, or cpu.",
        ),
    ] = "auto",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Only process the first N images for a smoke run.",
        ),
    ] = None,
    preview_dir: Annotated[
        Path | None,
        typer.Option(
            "--preview-dir",
            help="Optional directory for annotated preview images.",
        ),
    ] = None,
    max_previews: Annotated[
        int,
        typer.Option(
            "--max-previews",
            min=0,
            help="Maximum number of preview images to write when --preview-dir is set.",
        ),
    ] = 20,
) -> None:
    """Run RF-DETR on the local COCO benchmark split."""
    try:
        resolved_model_name = model_name or rfdetr_model_name_from_weights(weights)
        resolved_run_id = run_id or default_run_id()
        resolved_output_path = output_path or (
            DEFAULT_RUNS_DIR
            / "benchmarks"
            / resolved_run_id
            / "predictions"
            / f"{resolved_model_name}.jsonl"
        )
        result = predict_faces_from_coco_dataset(
            dataset_dir=dataset_dir,
            output_path=resolved_output_path,
            weights_path=weights,
            threshold=threshold,
            batch_size=batch_size,
            max_detections=max_detections,
            device=device,
            limit=limit,
            preview_dir=preview_dir,
            max_previews=max_previews,
            model_name=resolved_model_name,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Wrote detections for {result.image_count} benchmark images "
        f"({result.detection_count} boxes) to {result.output_path}"
    )
    if result.preview_count:
        typer.echo(f"Preview images: {result.preview_count} in {result.preview_dir}")
