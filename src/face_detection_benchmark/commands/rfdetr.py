"""RF-DETR prediction CLI command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from face_detection_benchmark.commands.common import (
    benchmark_latency_paths,
    default_run_id,
)
from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAMES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
    DEFAULT_RUNS_DIR,
    DEFAULT_TRAINING_RUNS_DIR,
)
from face_detection_benchmark.inference import (
    DEFAULT_VALIDATION_INFERENCE_THRESHOLD,
    predict_faces_from_coco_dataset,
    predict_faces_from_frames,
    rfdetr_model_name_from_weights,
)
from face_detection_benchmark.models.rfdetr import (
    RFDETR_DATASET_FILES,
    RfdetrTrainingConfig,
)
from face_detection_benchmark.models.rfdetr import (
    train_rfdetr as train_rfdetr_model,
)
from face_detection_benchmark.reports import (
    parse_rfdetr_training_run_spec,
    write_rfdetr_training_comparison_reports,
    write_rfdetr_training_report,
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
        latency_path, latency_table_path = benchmark_latency_paths(
            resolved_output_path,
            resolved_model_name,
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
            latency_path=latency_path,
            latency_table_path=latency_table_path,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Wrote detections for {result.image_count} benchmark images "
        f"({result.detection_count} boxes) to {result.output_path}"
    )
    if result.preview_count:
        typer.echo(f"Preview images: {result.preview_count} in {result.preview_dir}")
    if result.latency_path is not None:
        typer.echo(f"Latency summary: {result.latency_path}")


def train_rfdetr(
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            help=(
                "RF-DETR training dataset root. Must be explicit and must not be "
                "inside data/benchmark/."
            ),
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help=(
                "Training output directory. For example: "
                f"{DEFAULT_TRAINING_RUNS_DIR}/<run-id>."
            ),
        ),
    ],
    weights: Annotated[
        Path | None,
        typer.Option(
            "--weights",
            help="Optional local RF-DETR checkpoint to use as pretraining weights.",
        ),
    ] = None,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            min=1,
            help="Number of RF-DETR training epochs.",
        ),
    ] = 100,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="RF-DETR training batch size.",
        ),
    ] = 4,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device to use: auto, mps, cuda, or cpu.",
        ),
    ] = "auto",
    dataset_file: Annotated[
        str,
        typer.Option(
            "--dataset-file",
            help=f"RF-DETR dataset format: {', '.join(RFDETR_DATASET_FILES)}.",
        ),
    ] = "roboflow",
    num_workers: Annotated[
        int,
        typer.Option(
            "--num-workers",
            min=0,
            help="Number of RF-DETR training dataloader workers.",
        ),
    ] = 2,
) -> None:
    """Train RF-DETR on an explicit training dataset."""
    try:
        result = train_rfdetr_model(
            RfdetrTrainingConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                dataset_file=dataset_file,
                num_workers=num_workers,
                weights_path=weights,
            )
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(f"RF-DETR training output: {result.output_dir}")
    typer.echo(f"Config: {result.config_path}")
    typer.echo(f"Metadata: {result.metadata_path}")


def report_rfdetr_training(
    metrics_csv: Annotated[
        Path,
        typer.Option(
            "--metrics-csv",
            help="RF-DETR training metrics.csv path.",
        ),
    ],
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help=(
                "Training report output directory. Defaults to "
                "runs/training-reports/<run-id>."
            ),
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Run id used when --output-dir is not supplied.",
        ),
    ] = None,
    selection_metric: Annotated[
        str,
        typer.Option(
            "--selection-metric",
            help=("Best-row metric: f2, f1, precision, recall, map50, or map50-95."),
        ),
    ] = "f2",
) -> None:
    """Generate RF-DETR training metrics reports from metrics.csv."""
    try:
        resolved_run_id = run_id or metrics_csv.parent.name
        resolved_output_dir = output_dir or (
            DEFAULT_RUNS_DIR / "training-reports" / resolved_run_id
        )
        report_paths = write_rfdetr_training_report(
            metrics_csv_path=metrics_csv,
            output_dir=resolved_output_dir,
            run_id=resolved_run_id,
            selection_metric=selection_metric,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(f"RF-DETR training report: {resolved_output_dir}")
    typer.echo(f"Clean metrics CSV: {report_paths['metrics_clean_path']}")
    typer.echo(f"Metrics Markdown: {report_paths['metrics_markdown_path']}")
    typer.echo(f"Summary Markdown: {report_paths['summary_path']}")
    typer.echo(f"Loss plot: {report_paths['loss_path']}")
    typer.echo(f"Validation score plot: {report_paths['score_path']}")
    typer.echo(f"mAP plot: {report_paths['map_path']}")
    if "learning_rate_path" in report_paths:
        typer.echo(f"Learning-rate plot: {report_paths['learning_rate_path']}")


def compare_rfdetr_training_runs(
    training_runs: Annotated[
        list[str] | None,
        typer.Option(
            "--training-run",
            help=(
                "RF-DETR training report as optional display label and path, "
                "for example EMA1=runs/training-reports/run-a. Plain paths are "
                "also supported. Repeat this option for each run."
            ),
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help=(
                "Comparison output directory. Defaults to "
                "runs/training-reports/comparisons/<run-id>."
            ),
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Run id used when --output-dir is not supplied.",
        ),
    ] = None,
    selection_metric: Annotated[
        str,
        typer.Option(
            "--selection-metric",
            help=(
                "Comparison ranking metric: f2, f1, precision, recall, map50, "
                "or map50-95."
            ),
        ),
    ] = "f2",
) -> None:
    """Compare RF-DETR training reports on shared plots."""
    try:
        resolved_training_runs = [
            parse_rfdetr_training_run_spec(training_run)
            for training_run in training_runs or []
        ]
        resolved_run_id = run_id or default_run_id()
        resolved_output_dir = output_dir or (
            DEFAULT_RUNS_DIR / "training-reports" / "comparisons" / resolved_run_id
        )
        report_paths = write_rfdetr_training_comparison_reports(
            training_run_specs=resolved_training_runs,
            output_dir=resolved_output_dir,
            selection_metric=selection_metric,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Compared {len(resolved_training_runs)} RF-DETR training runs in "
        f"{resolved_output_dir}"
    )
    typer.echo(f"Summary CSV: {report_paths['summary_csv_path']}")
    typer.echo(f"Summary Markdown: {report_paths['summary_markdown_path']}")
    typer.echo(f"Validation F2 overlay: {report_paths['validation_f2_path']}")
    typer.echo(f"mAP overlay: {report_paths['map_path']}")
    typer.echo(f"Loss overlay: {report_paths['loss_path']}")
    if "learning_rate_path" in report_paths:
        typer.echo(f"Learning-rate overlay: {report_paths['learning_rate_path']}")
