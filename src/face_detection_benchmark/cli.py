"""Command-line entrypoints for the face detection benchmark pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from face_detection_benchmark.coco import export_predictions_to_coco
from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_BENCHMARK_IMAGE_COUNT,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_FPS,
    DEFAULT_FRAMES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_ROBOFLOW_EXPORT_DIR,
    DEFAULT_ROBOFLOW_FORMAT,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
    DEFAULT_RUNS_DIR,
    DEFAULT_VIDEO_DIR,
    FACE_CATEGORY_NAME,
)
from face_detection_benchmark.datasets import download_roboflow_benchmark_dataset
from face_detection_benchmark.env import get_env_value
from face_detection_benchmark.evaluation import (
    DEFAULT_SWEEP_THRESHOLDS,
    evaluate_coco_predictions,
    evaluate_confidence_thresholds,
)
from face_detection_benchmark.inference import (
    DEFAULT_VALIDATION_INFERENCE_THRESHOLD,
    predict_faces_from_coco_dataset,
    predict_faces_from_frames,
    rfdetr_model_name_from_weights,
)
from face_detection_benchmark.reports import (
    write_evaluation_reports,
    write_threshold_validation_reports,
)
from face_detection_benchmark.video import extract_video_frames

app = typer.Typer(
    help="Extract frames, run face detection models, and export benchmark datasets."
)


@app.command()
def extract_frames(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory containing source MP4 videos.",
        ),
    ] = DEFAULT_VIDEO_DIR,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where extracted frames and metadata are written.",
        ),
    ] = DEFAULT_FRAMES_DIR,
    fps: Annotated[
        float,
        typer.Option(
            "--fps",
            min=0.001,
            help="Sample this many frames per second.",
        ),
    ] = DEFAULT_FRAME_FPS,
    every_n_frames: Annotated[
        int | None,
        typer.Option(
            "--every-n-frames",
            min=1,
            help="Sample by source frame interval instead of seconds.",
        ),
    ] = None,
    image_format: Annotated[
        str,
        typer.Option(
            "--image-format",
            help="Image format for extracted frames: jpg, jpeg, or png.",
        ),
    ] = "jpg",
    quality: Annotated[
        int,
        typer.Option(
            "--quality",
            min=1,
            max=100,
            help="JPEG quality when writing jpg/jpeg frames.",
        ),
    ] = 95,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite existing extracted frame images.",
        ),
    ] = False,
) -> None:
    """Extract sampled frames from the source videos."""
    try:
        result = extract_video_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            fps=fps,
            every_n_frames=every_n_frames,
            image_format=image_format,
            quality=quality,
            overwrite=overwrite,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Extracted {result.frame_count} frames from {result.video_count} videos "
        f"to {result.output_dir}"
    )
    typer.echo(f"Metadata: {result.metadata_path}")


@app.command()
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


@app.command()
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
        resolved_run_id = run_id or _default_run_id()
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


@app.command()
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


@app.command()
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
        resolved_version = version or _optional_int_env("ROBOFLOW_VERSION")
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


@app.command()
def evaluate_detections(
    predictions_path: Annotated[
        Path,
        typer.Option(
            "--predictions-path",
            help="Normalized prediction JSONL to evaluate.",
        ),
    ],
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            help="COCO split directory containing _annotations.coco.json.",
        ),
    ] = DEFAULT_BENCHMARK_DATA_DIR
    / DEFAULT_BENCHMARK_DATASET_NAME
    / DEFAULT_ROBOFLOW_TEST_SPLIT,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Benchmark run output directory. Defaults to runs/benchmarks/<run-id>.",
        ),
    ] = None,
    results_table_path: Annotated[
        Path,
        typer.Option(
            "--results-table",
            help="Cumulative local CSV table appended after each evaluation.",
        ),
    ] = DEFAULT_RUNS_DIR / "benchmarks" / "results.csv",
    leaderboard_path: Annotated[
        Path,
        typer.Option(
            "--leaderboard",
            help="Human-readable Markdown leaderboard generated from the results table.",
        ),
    ] = DEFAULT_RUNS_DIR / "benchmarks" / "results.md",
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Run id used when --output-dir is not supplied.",
        ),
    ] = None,
    category_name: Annotated[
        str,
        typer.Option(
            "--category",
            help="COCO category name to evaluate.",
        ),
    ] = FACE_CATEGORY_NAME,
    confidence_threshold: Annotated[
        float | None,
        typer.Option(
            "--confidence-threshold",
            min=0.0,
            max=1.0,
            help="Required preselected confidence threshold for precision/recall/F1/F2.",
        ),
    ] = None,
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou-threshold",
            min=0.0,
            max=1.0,
            help="IoU threshold for primary precision/recall/F1.",
        ),
    ] = 0.5,
    include_confidence_sweep: Annotated[
        bool,
        typer.Option(
            "--include-confidence-sweep",
            help=(
                "Write diagnostic confidence sweep output. Do not use this to "
                "choose thresholds on the benchmark test set."
            ),
        ),
    ] = False,
) -> None:
    """Evaluate normalized detections against a COCO benchmark split."""
    try:
        if confidence_threshold is None:
            raise ValueError(
                "--confidence-threshold is required; choose it before evaluating "
                "the benchmark"
            )
        metrics = evaluate_coco_predictions(
            dataset_dir=dataset_dir,
            predictions_path=predictions_path,
            category_name=category_name,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            sweep_thresholds=(
                DEFAULT_SWEEP_THRESHOLDS if include_confidence_sweep else None
            ),
        )
        resolved_run_id = run_id or _default_run_id()
        resolved_output_dir = output_dir or (
            DEFAULT_RUNS_DIR / "benchmarks" / resolved_run_id
        )
        report_paths = write_evaluation_reports(
            metrics,
            resolved_output_dir,
            results_table_path=results_table_path,
            leaderboard_path=leaderboard_path,
            run_id=resolved_run_id,
            dataset_dir=dataset_dir,
            predictions_path=predictions_path,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Evaluated {metrics.prediction_count} predictions against "
        f"{metrics.ground_truth_count} {category_name} boxes"
    )
    typer.echo(
        f"precision={metrics.precision:.4f} recall={metrics.recall:.4f} "
        f"f1={metrics.f1:.4f} f2={metrics.f2:.4f} "
        f"AP50={metrics.ap50:.4f} "
        f"mAP50-95={metrics.map_50_95:.4f}"
    )
    typer.echo(f"Metrics JSON: {report_paths['metrics_path']}")
    typer.echo(f"Summary CSV: {report_paths['summary_path']}")
    typer.echo(f"Results table: {report_paths['results_table_path']}")
    typer.echo(f"Leaderboard: {report_paths['leaderboard_path']}")
    if "sweep_path" in report_paths:
        typer.echo(f"Confidence sweep CSV: {report_paths['sweep_path']}")


@app.command()
def validate_thresholds(
    predictions_path: Annotated[
        Path,
        typer.Option(
            "--predictions-path",
            help="Normalized prediction JSONL to evaluate across thresholds.",
        ),
    ],
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            help="COCO validation split directory containing _annotations.coco.json.",
        ),
    ] = DEFAULT_BENCHMARK_DATA_DIR
    / DEFAULT_BENCHMARK_DATASET_NAME
    / DEFAULT_ROBOFLOW_TEST_SPLIT,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help="Validation output directory. Defaults to runs/validation/<run-id>.",
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Run id used when --output-dir is not supplied.",
        ),
    ] = None,
    category_name: Annotated[
        str,
        typer.Option(
            "--category",
            help="COCO category name to evaluate.",
        ),
    ] = FACE_CATEGORY_NAME,
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou-threshold",
            min=0.0,
            max=1.0,
            help="IoU threshold for precision/recall/F-score validation.",
        ),
    ] = 0.5,
    selection_metric: Annotated[
        str,
        typer.Option(
            "--selection-metric",
            help="Metric used to select the threshold: precision, recall, f1, or f2.",
        ),
    ] = "f2",
    thresholds: Annotated[
        str | None,
        typer.Option(
            "--thresholds",
            help="Comma-separated thresholds. Defaults to 0.005,0.01,0.05,...,0.80.",
        ),
    ] = None,
) -> None:
    """Treat a labeled split as validation and choose a confidence threshold."""
    try:
        threshold_values = (
            _parse_thresholds(thresholds)
            if thresholds is not None
            else DEFAULT_SWEEP_THRESHOLDS
        )
        result = evaluate_confidence_thresholds(
            dataset_dir=dataset_dir,
            predictions_path=predictions_path,
            category_name=category_name,
            iou_threshold=iou_threshold,
            thresholds=threshold_values,
            selection_metric=selection_metric,
        )
        resolved_run_id = run_id or _default_run_id()
        resolved_output_dir = output_dir or (
            DEFAULT_RUNS_DIR / "validation" / resolved_run_id
        )
        report_paths = write_threshold_validation_reports(
            result=result,
            output_dir=resolved_output_dir,
            dataset_dir=dataset_dir,
            predictions_path=predictions_path,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Validated {len(result.threshold_metrics)} thresholds for "
        f"{result.model_name} against {result.ground_truth_count} {category_name} boxes"
    )
    typer.echo(
        f"selected_threshold={result.selected_threshold:.4f} "
        f"selection_metric={result.selection_metric} "
        f"precision={float(result.selected_metrics['precision']):.4f} "
        f"recall={float(result.selected_metrics['recall']):.4f} "
        f"f1={float(result.selected_metrics['f1']):.4f} "
        f"f2={float(result.selected_metrics['f2']):.4f}"
    )
    typer.echo(f"Validation JSON: {report_paths['validation_path']}")
    typer.echo(f"Selected threshold JSON: {report_paths['selected_threshold_path']}")
    typer.echo(f"Threshold CSV: {report_paths['threshold_metrics_path']}")
    typer.echo(f"Threshold Markdown: {report_paths['threshold_metrics_markdown_path']}")
    typer.echo(f"Precision-recall plot: {report_paths['precision_recall_path']}")
    typer.echo(f"F-score plot: {report_paths['f_scores_path']}")


@app.command()
def upload_roboflow() -> None:
    """Upload the exported COCO dataset to Roboflow."""
    typer.echo("upload-roboflow is planned for a later checkpoint.")


def _optional_int_env(name: str) -> int | None:
    value = get_env_value(name)
    if not value:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_thresholds(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--thresholds must be a comma-separated list of numbers")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as error:
        raise ValueError("--thresholds must contain only numbers") from error
