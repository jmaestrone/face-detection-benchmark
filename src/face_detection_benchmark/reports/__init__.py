"""Report writers for benchmark evaluation outputs."""

import json
from pathlib import Path

from face_detection_benchmark.evaluation import (
    metrics_to_json_dict,
    threshold_validation_to_json_dict,
)
from face_detection_benchmark.evaluation.types import (
    DetectionMetrics,
    ThresholdValidationResult,
)
from face_detection_benchmark.reports.charts import (
    write_f_scores_overlay_svg,
    write_f_scores_svg,
    write_precision_recall_overlay_svg,
    write_precision_recall_svg,
)
from face_detection_benchmark.reports.comparison import (
    ValidationRunSpec,
    load_validation_runs,
    parse_validation_run_spec,
    write_validation_comparison_reports,
)
from face_detection_benchmark.reports.overlays import (
    PredictionOverlayResult,
    PredictionOverlaySpec,
    parse_prediction_overlay_spec,
    render_prediction_overlays,
)
from face_detection_benchmark.reports.tables import (
    append_results_row,
    write_results_leaderboard,
    write_summary_csv,
    write_sweep_csv,
    write_threshold_metrics_csv,
    write_threshold_metrics_markdown,
)

__all__ = [
    "append_results_row",
    "parse_prediction_overlay_spec",
    "PredictionOverlayResult",
    "PredictionOverlaySpec",
    "render_prediction_overlays",
    "write_evaluation_reports",
    "write_f_scores_overlay_svg",
    "write_f_scores_svg",
    "write_precision_recall_overlay_svg",
    "write_precision_recall_svg",
    "write_results_leaderboard",
    "write_summary_csv",
    "write_sweep_csv",
    "write_threshold_metrics_csv",
    "write_threshold_metrics_markdown",
    "write_threshold_validation_reports",
    "ValidationRunSpec",
    "load_validation_runs",
    "parse_validation_run_spec",
    "write_validation_comparison_reports",
]


def write_evaluation_reports(
    metrics: DetectionMetrics,
    output_dir: Path,
    results_table_path: Path | None = None,
    leaderboard_path: Path | None = None,
    run_id: str | None = None,
    dataset_dir: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Path]:
    """Write per-run reports and optionally append a cumulative results row."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{metrics.model_name}.json"
    summary_path = output_dir / "summary.csv"
    sweep_path = metrics_dir / f"{metrics.model_name}_confidence_sweep.csv"

    metrics_path.write_text(
        json.dumps(metrics_to_json_dict(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary_csv(metrics, summary_path)

    report_paths = {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
    }
    if metrics.confidence_sweep:
        write_sweep_csv(metrics, sweep_path)
        report_paths["sweep_path"] = sweep_path
    if results_table_path is not None:
        append_results_row(
            metrics=metrics,
            results_table_path=results_table_path,
            run_id=run_id,
            dataset_dir=dataset_dir,
            predictions_path=predictions_path,
        )
        report_paths["results_table_path"] = results_table_path
        if leaderboard_path is not None:
            write_results_leaderboard(
                results_table_path=results_table_path,
                leaderboard_path=leaderboard_path,
            )
            report_paths["leaderboard_path"] = leaderboard_path
    return report_paths


def write_threshold_validation_reports(
    result: ThresholdValidationResult,
    output_dir: Path,
    dataset_dir: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Path]:
    """Write validation threshold tables, selection metadata, and SVG plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "threshold_validation.json"
    selected_threshold_path = output_dir / "selected_threshold.json"
    threshold_metrics_path = output_dir / "threshold_metrics.csv"
    threshold_metrics_markdown_path = output_dir / "threshold_metrics.md"
    precision_recall_path = output_dir / "precision_recall.svg"
    f_scores_path = output_dir / "f1_f2_by_threshold.svg"

    validation_payload = threshold_validation_to_json_dict(result)
    validation_payload["dataset_dir"] = dataset_dir.as_posix() if dataset_dir else ""
    validation_payload["predictions_path"] = (
        predictions_path.as_posix() if predictions_path else ""
    )
    validation_path.write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    selected_threshold_path.write_text(
        json.dumps(
            {
                "model_name": result.model_name,
                "selection_metric": result.selection_metric,
                "selected_threshold": result.selected_threshold,
                "selected_metrics": result.selected_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_threshold_metrics_csv(result, threshold_metrics_path)
    write_threshold_metrics_markdown(result, threshold_metrics_markdown_path)
    write_precision_recall_svg(result, precision_recall_path)
    write_f_scores_svg(result, f_scores_path)
    return {
        "validation_path": validation_path,
        "selected_threshold_path": selected_threshold_path,
        "threshold_metrics_path": threshold_metrics_path,
        "threshold_metrics_markdown_path": threshold_metrics_markdown_path,
        "precision_recall_path": precision_recall_path,
        "f_scores_path": f_scores_path,
    }
