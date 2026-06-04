"""Report writers for benchmark evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from face_detection_benchmark.charts import (
    write_f_scores_svg,
    write_precision_recall_svg,
)
from face_detection_benchmark.evaluation import (
    DetectionMetrics,
    ThresholdValidationResult,
    metrics_to_json_dict,
    threshold_validation_to_json_dict,
)


def _read_results_rows(results_table_path: Path) -> list[dict[str, Any]]:
    """Read cumulative benchmark result rows from CSV."""
    if not results_table_path.exists():
        return []
    with results_table_path.open("r", encoding="utf-8", newline="") as results_file:
        return list(csv.DictReader(results_file))


def _leaderboard_lines(rows: list[dict[str, Any]]) -> list[str]:
    """Format leaderboard rows as Markdown table lines."""
    lines = [
        "# Face Detection Benchmark Results",
        "",
        "Sorted by `mAP@[0.50:0.95]`, then F2, then AP50.",
        "",
        "| Rank | Run | Model | Threshold | Precision | Recall | F1 | F2 | AP50 | "
        "AP75 | mAP50-95 | TP | FP | FN |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for rank, results_row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(results_row.get("run_id", "")),
                    str(results_row.get("model_name", "")),
                    _format_float(results_row.get("confidence_threshold", "")),
                    _format_float(results_row.get("precision", "")),
                    _format_float(results_row.get("recall", "")),
                    _format_float(results_row.get("f1", "")),
                    _format_float(results_row.get("f2", "")),
                    _format_float(results_row.get("ap50", "")),
                    _format_float(results_row.get("ap75", "")),
                    _format_float(results_row.get("map_50_95", "")),
                    str(results_row.get("true_positive_count", "")),
                    str(results_row.get("false_positive_count", "")),
                    str(results_row.get("false_negative_count", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "This file is generated from `runs/benchmarks/results.csv` and is "
            "ignored by git.",
        ]
    )
    return lines


def _float_value(row: dict[str, Any], key: str) -> float:
    """Read a sortable float from a CSV row."""
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def _format_float(value: Any) -> str:
    """Format table floats with four decimal places."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _write_summary_csv(metrics: DetectionMetrics, summary_path: Path) -> None:
    """Write the per-run summary CSV."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "model_name",
                "image_count",
                "ground_truth_count",
                "prediction_count",
                "confidence_threshold",
                "iou_threshold",
                "true_positive_count",
                "false_positive_count",
                "false_negative_count",
                "precision",
                "recall",
                "f1",
                "f2",
                "ap50",
                "ap75",
                "map_50_95",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_name": metrics.model_name,
                "image_count": metrics.image_count,
                "ground_truth_count": metrics.ground_truth_count,
                "prediction_count": metrics.prediction_count,
                "confidence_threshold": metrics.confidence_threshold,
                "iou_threshold": metrics.iou_threshold,
                "true_positive_count": metrics.true_positive_count,
                "false_positive_count": metrics.false_positive_count,
                "false_negative_count": metrics.false_negative_count,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "f2": metrics.f2,
                "ap50": metrics.ap50,
                "ap75": metrics.ap75,
                "map_50_95": metrics.map_50_95,
            }
        )


def _write_sweep_csv(metrics: DetectionMetrics, sweep_path: Path) -> None:
    """Write the confidence sweep CSV when metrics include one."""
    with sweep_path.open("w", encoding="utf-8", newline="") as sweep_file:
        fieldnames = [
            "confidence_threshold",
            "true_positive_count",
            "false_positive_count",
            "false_negative_count",
            "precision",
            "recall",
            "f1",
            "f2",
        ]
        writer = csv.DictWriter(sweep_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics.confidence_sweep)


def _write_threshold_metrics_csv(
    result: ThresholdValidationResult,
    threshold_metrics_path: Path,
) -> None:
    """Write validation threshold metrics as CSV."""
    threshold_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with threshold_metrics_path.open(
        "w", encoding="utf-8", newline=""
    ) as threshold_metrics_file:
        writer = csv.DictWriter(
            threshold_metrics_file,
            fieldnames=[
                "confidence_threshold",
                "true_positive_count",
                "false_positive_count",
                "false_negative_count",
                "precision",
                "recall",
                "f1",
                "f2",
            ],
        )
        writer.writeheader()
        writer.writerows(result.threshold_metrics)


def _write_threshold_metrics_markdown(
    result: ThresholdValidationResult,
    threshold_metrics_markdown_path: Path,
) -> None:
    """Write validation threshold metrics as Markdown."""
    lines = [
        "# Validation Threshold Metrics",
        "",
        f"Model: `{result.model_name}`",
        f"Selection metric: `{result.selection_metric}`",
        f"Selected threshold: `{result.selected_threshold:.4f}`",
        f"IoU threshold: `{result.iou_threshold:.4f}`",
        "",
        "| Threshold | Precision | Recall | F1 | F2 | TP | FP | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for threshold_row in result.threshold_metrics:
        prefix = "**" if threshold_row == result.selected_metrics else ""
        suffix = "**" if threshold_row == result.selected_metrics else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    (
                        f"{prefix}"
                        f"{float(threshold_row['confidence_threshold']):.4f}"
                        f"{suffix}"
                    ),
                    f"{prefix}{float(threshold_row['precision']):.4f}{suffix}",
                    f"{prefix}{float(threshold_row['recall']):.4f}{suffix}",
                    f"{prefix}{float(threshold_row['f1']):.4f}{suffix}",
                    f"{prefix}{float(threshold_row['f2']):.4f}{suffix}",
                    f"{prefix}{int(threshold_row['true_positive_count'])}{suffix}",
                    (f"{prefix}{int(threshold_row['false_positive_count'])}{suffix}"),
                    (f"{prefix}{int(threshold_row['false_negative_count'])}{suffix}"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "This validation output can be used to choose an operating threshold. "
            "Do not also report the same data as an unbiased test benchmark.",
        ]
    )
    threshold_metrics_markdown_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


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
    _write_summary_csv(metrics, summary_path)

    report_paths = {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
    }
    if metrics.confidence_sweep:
        _write_sweep_csv(metrics, sweep_path)
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
    _write_threshold_metrics_csv(result, threshold_metrics_path)
    _write_threshold_metrics_markdown(result, threshold_metrics_markdown_path)
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


def append_results_row(
    metrics: DetectionMetrics,
    results_table_path: Path,
    run_id: str | None,
    dataset_dir: Path | None,
    predictions_path: Path | None,
) -> None:
    """Append one evaluation row to the cumulative local results table."""
    results_table_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "model_name",
        "dataset_dir",
        "predictions_path",
        "image_count",
        "ground_truth_count",
        "prediction_count",
        "confidence_threshold",
        "iou_threshold",
        "true_positive_count",
        "false_positive_count",
        "false_negative_count",
        "precision",
        "recall",
        "f1",
        "f2",
        "ap50",
        "ap75",
        "map_50_95",
    ]
    should_write_header = not results_table_path.exists()
    with results_table_path.open("a", encoding="utf-8", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id or "",
                "model_name": metrics.model_name,
                "dataset_dir": dataset_dir.as_posix() if dataset_dir else "",
                "predictions_path": (
                    predictions_path.as_posix() if predictions_path else ""
                ),
                "image_count": metrics.image_count,
                "ground_truth_count": metrics.ground_truth_count,
                "prediction_count": metrics.prediction_count,
                "confidence_threshold": metrics.confidence_threshold,
                "iou_threshold": metrics.iou_threshold,
                "true_positive_count": metrics.true_positive_count,
                "false_positive_count": metrics.false_positive_count,
                "false_negative_count": metrics.false_negative_count,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "f2": metrics.f2,
                "ap50": metrics.ap50,
                "ap75": metrics.ap75,
                "map_50_95": metrics.map_50_95,
            }
        )


def write_results_leaderboard(
    results_table_path: Path,
    leaderboard_path: Path,
) -> None:
    """Write a human-readable Markdown leaderboard from the results CSV."""
    rows = _read_results_rows(results_table_path)
    sorted_rows = sorted(
        rows,
        key=lambda results_row: (
            _float_value(results_row, "map_50_95"),
            _float_value(results_row, "f2"),
            _float_value(results_row, "ap50"),
        ),
        reverse=True,
    )
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(
        "\n".join(_leaderboard_lines(sorted_rows)) + "\n",
        encoding="utf-8",
    )
