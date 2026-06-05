"""Comparison reports for multiple threshold validation runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from face_detection_benchmark.evaluation.types import ThresholdValidationResult
from face_detection_benchmark.reports.charts import (
    write_f_scores_overlay_svg,
    write_precision_recall_overlay_svg,
)

THRESHOLD_VALIDATION_FILE_NAME = "threshold_validation.json"


@dataclass(frozen=True)
class ValidationRun:
    """One loaded threshold validation run and its source path."""

    run_id: str
    source_path: Path
    result: ThresholdValidationResult


def write_validation_comparison_reports(
    validation_run_paths: list[Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Write summary tables and overlay plots for validation runs."""
    validation_runs = load_validation_runs(validation_run_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "summary.csv"
    summary_markdown_path = output_dir / "summary.md"
    precision_recall_path = output_dir / "precision_recall_overlay.svg"
    f_scores_path = output_dir / "f1_f2_overlay.svg"

    write_validation_comparison_csv(validation_runs, summary_csv_path)
    write_validation_comparison_markdown(validation_runs, summary_markdown_path)
    write_precision_recall_overlay_svg(
        [validation_run.result for validation_run in validation_runs],
        precision_recall_path,
    )
    write_f_scores_overlay_svg(
        [validation_run.result for validation_run in validation_runs],
        f_scores_path,
    )
    return {
        "summary_csv_path": summary_csv_path,
        "summary_markdown_path": summary_markdown_path,
        "precision_recall_path": precision_recall_path,
        "f_scores_path": f_scores_path,
    }


def load_validation_runs(validation_run_paths: list[Path]) -> list[ValidationRun]:
    """Load threshold validation results from run directories or JSON paths."""
    if len(validation_run_paths) < 2:
        raise ValueError("At least two --validation-run values are required")

    validation_runs = [
        _load_validation_run(validation_run_path)
        for validation_run_path in validation_run_paths
    ]
    iou_thresholds = {
        validation_run.result.iou_threshold for validation_run in validation_runs
    }
    if len(iou_thresholds) > 1:
        raise ValueError("Validation runs must use the same IoU threshold")
    return validation_runs


def write_validation_comparison_csv(
    validation_runs: list[ValidationRun],
    summary_csv_path: Path,
) -> None:
    """Write selected operating-point metrics for multiple validation runs."""
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as summary_file:
        fieldnames = [
            "rank",
            "run_id",
            "model_name",
            "selection_metric",
            "selected_threshold",
            "precision",
            "recall",
            "f1",
            "f2",
            "true_positive_count",
            "false_positive_count",
            "false_negative_count",
            "image_count",
            "ground_truth_count",
            "prediction_count",
            "source_path",
        ]
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, validation_run in enumerate(
            _rank_validation_runs(validation_runs),
            start=1,
        ):
            writer.writerow(_validation_summary_row(rank, validation_run))


def write_validation_comparison_markdown(
    validation_runs: list[ValidationRun],
    summary_markdown_path: Path,
) -> None:
    """Write selected operating-point metrics as a Markdown table."""
    lines = [
        "# Validation Run Comparison",
        "",
        "Sorted by each run's selected metric value, then F2, then F1.",
        "",
        "| Rank | Run | Model | Metric | Threshold | Precision | Recall | F1 | F2 | "
        "TP | FP | FN |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for rank, validation_run in enumerate(
        _rank_validation_runs(validation_runs),
        start=1,
    ):
        selected_metrics = validation_run.result.selected_metrics
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    validation_run.run_id,
                    validation_run.result.model_name,
                    validation_run.result.selection_metric,
                    _format_float(validation_run.result.selected_threshold),
                    _format_float(selected_metrics["precision"]),
                    _format_float(selected_metrics["recall"]),
                    _format_float(selected_metrics["f1"]),
                    _format_float(selected_metrics["f2"]),
                    str(int(selected_metrics["true_positive_count"])),
                    str(int(selected_metrics["false_positive_count"])),
                    str(int(selected_metrics["false_negative_count"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "This comparison is generated from validation runs and is ignored by git.",
        ]
    )
    summary_markdown_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _load_validation_run(validation_run_path: Path) -> ValidationRun:
    validation_path = _validation_json_path(validation_run_path)
    if not validation_path.exists():
        raise ValueError(f"Validation JSON does not exist: {validation_path}")
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    return ValidationRun(
        run_id=validation_path.parent.name,
        source_path=validation_path,
        result=ThresholdValidationResult(
            model_name=str(payload["model_name"]),
            image_count=int(payload["image_count"]),
            ground_truth_count=int(payload["ground_truth_count"]),
            prediction_count=int(payload["prediction_count"]),
            iou_threshold=float(payload["iou_threshold"]),
            selection_metric=str(payload["selection_metric"]),
            selected_threshold=float(payload["selected_threshold"]),
            selected_metrics=dict(payload["selected_metrics"]),
            threshold_metrics=list(payload["threshold_metrics"]),
        ),
    )


def _validation_json_path(validation_run_path: Path) -> Path:
    if validation_run_path.is_dir():
        return validation_run_path / THRESHOLD_VALIDATION_FILE_NAME
    return validation_run_path


def _rank_validation_runs(validation_runs: list[ValidationRun]) -> list[ValidationRun]:
    return sorted(
        validation_runs,
        key=lambda validation_run: (
            float(
                validation_run.result.selected_metrics[
                    validation_run.result.selection_metric
                ]
            ),
            float(validation_run.result.selected_metrics["f2"]),
            float(validation_run.result.selected_metrics["f1"]),
        ),
        reverse=True,
    )


def _validation_summary_row(
    rank: int,
    validation_run: ValidationRun,
) -> dict[str, Any]:
    selected_metrics = validation_run.result.selected_metrics
    return {
        "rank": rank,
        "run_id": validation_run.run_id,
        "model_name": validation_run.result.model_name,
        "selection_metric": validation_run.result.selection_metric,
        "selected_threshold": validation_run.result.selected_threshold,
        "precision": selected_metrics["precision"],
        "recall": selected_metrics["recall"],
        "f1": selected_metrics["f1"],
        "f2": selected_metrics["f2"],
        "true_positive_count": selected_metrics["true_positive_count"],
        "false_positive_count": selected_metrics["false_positive_count"],
        "false_negative_count": selected_metrics["false_negative_count"],
        "image_count": validation_run.result.image_count,
        "ground_truth_count": validation_run.result.ground_truth_count,
        "prediction_count": validation_run.result.prediction_count,
        "source_path": validation_run.source_path.as_posix(),
    }


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""
