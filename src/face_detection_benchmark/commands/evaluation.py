"""Evaluation CLI command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from face_detection_benchmark.commands.common import default_run_id, parse_thresholds
from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
    DEFAULT_RUNS_DIR,
    FACE_CATEGORY_NAME,
)
from face_detection_benchmark.evaluation import (
    DEFAULT_SWEEP_THRESHOLDS,
    evaluate_coco_predictions,
    evaluate_confidence_thresholds,
)
from face_detection_benchmark.reports import (
    parse_prediction_overlay_spec,
    write_evaluation_reports,
    write_threshold_validation_reports,
    write_validation_comparison_reports,
)
from face_detection_benchmark.reports import (
    render_prediction_overlays as render_prediction_overlay_reports,
)


def evaluate_detections(
    predictions_path: Annotated[
        Path,
        typer.Option(
            "--predictions-path",
            help="Normalized prediction JSONL to evaluate.",
        ),
    ],
    confidence_threshold: Annotated[
        float,
        typer.Option(
            "--confidence-threshold",
            min=0.0,
            max=1.0,
            help=(
                "Required preselected confidence threshold for precision/recall/F1/F2."
            ),
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
            help=(
                "Benchmark run output directory. Defaults to runs/benchmarks/<run-id>."
            ),
        ),
    ] = None,
    results_table_path: Annotated[
        Path,
        typer.Option(
            "--results-table",
            help=("Cumulative local CSV table appended after each evaluation."),
        ),
    ] = DEFAULT_RUNS_DIR / "benchmarks" / "results.csv",
    leaderboard_path: Annotated[
        Path,
        typer.Option(
            "--leaderboard",
            help=(
                "Human-readable Markdown leaderboard generated from the results table."
            ),
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
        resolved_run_id = run_id or default_run_id()
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
            parse_thresholds(thresholds)
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
        resolved_run_id = run_id or default_run_id()
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


def compare_validation_runs(
    validation_runs: Annotated[
        list[Path] | None,
        typer.Option(
            "--validation-run",
            help=(
                "Validation run directory or threshold_validation.json path. "
                "Repeat this option for each model."
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
                "runs/validation/comparisons/<run-id>."
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
) -> None:
    """Compare multiple threshold validation runs on shared plots."""
    try:
        resolved_validation_runs = validation_runs or []
        resolved_run_id = run_id or default_run_id()
        resolved_output_dir = output_dir or (
            DEFAULT_RUNS_DIR / "validation" / "comparisons" / resolved_run_id
        )
        report_paths = write_validation_comparison_reports(
            validation_run_paths=resolved_validation_runs,
            output_dir=resolved_output_dir,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Compared {len(resolved_validation_runs)} validation runs in "
        f"{resolved_output_dir}"
    )
    typer.echo(f"Summary CSV: {report_paths['summary_csv_path']}")
    typer.echo(f"Summary Markdown: {report_paths['summary_markdown_path']}")
    typer.echo(f"Precision-recall overlay: {report_paths['precision_recall_path']}")
    typer.echo(f"F-score overlay: {report_paths['f_scores_path']}")


def render_prediction_overlays(
    prediction_specs: Annotated[
        list[str] | None,
        typer.Option(
            "--prediction-spec",
            help=(
                "Prediction overlay spec as label=path/to/predictions.jsonl:threshold. "
                "Repeat this option for each model."
            ),
        ),
    ] = None,
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
            help=(
                "Visualization output directory. Defaults to "
                "runs/visualizations/<run-id>."
            ),
        ),
    ] = None,
    runs_dir: Annotated[
        Path,
        typer.Option(
            "--runs-dir",
            help="Root runs directory used when --output-dir is not supplied.",
        ),
    ] = DEFAULT_RUNS_DIR,
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
            help="COCO category name to render.",
        ),
    ] = FACE_CATEGORY_NAME,
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou-threshold",
            min=0.0,
            max=1.0,
            help="IoU threshold for TP/FP/FN overlay classification.",
        ),
    ] = 0.5,
) -> None:
    """Render TP/FP/FN prediction overlay images for one or more models."""
    try:
        resolved_prediction_specs = [
            parse_prediction_overlay_spec(prediction_spec)
            for prediction_spec in prediction_specs or []
        ]
        resolved_run_id = run_id or default_run_id()
        resolved_output_dir = output_dir or (
            runs_dir / "visualizations" / resolved_run_id
        )
        result = render_prediction_overlay_reports(
            dataset_dir=dataset_dir,
            prediction_specs=resolved_prediction_specs,
            output_dir=resolved_output_dir,
            category_name=category_name,
            iou_threshold=iou_threshold,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Rendered overlays for {len(result.model_dirs)} model(s) and "
        f"{result.image_count} image(s)"
    )
    typer.echo(f"Output directory: {result.output_dir}")
    for label, model_dir in result.model_dirs.items():
        typer.echo(f"{label} overlays: {model_dir}")
    if result.comparison_dir is not None:
        typer.echo(f"Comparison overlays: {result.comparison_dir}")
    typer.echo(f"Summary CSV: {result.summary_csv_path}")
    typer.echo(f"Summary JSON: {result.summary_json_path}")
