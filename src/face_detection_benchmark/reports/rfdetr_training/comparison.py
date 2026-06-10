"""Comparison reports for multiple RF-DETR training runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from face_detection_benchmark.reports.rfdetr_training.charts import (
    write_runs_overlay_chart,
)
from face_detection_benchmark.reports.rfdetr_training.parsing import (
    format_float,
    lr_columns,
    read_rfdetr_clean_metrics,
    select_best_rfdetr_training_row,
)
from face_detection_benchmark.reports.rfdetr_training.types import (
    RfdetrTrainingRun,
    RfdetrTrainingRunSpec,
)


def _metrics_clean_path(report_path: Path) -> Path:
    """Return the cleaned metrics path for a report directory."""
    return report_path / "metrics_clean.csv"


def _rank_training_runs(
    training_runs: list[RfdetrTrainingRun],
) -> list[RfdetrTrainingRun]:
    """Rank RF-DETR training runs by selected metric and later progress."""
    return sorted(
        training_runs,
        key=lambda training_run: (
            float(training_run.best_row[training_run.selection_column]),
            int(training_run.best_row["epoch"]),
            int(training_run.best_row["step"]),
        ),
        reverse=True,
    )


def _comparison_summary_row(
    rank: int,
    training_run: RfdetrTrainingRun,
) -> dict[str, Any]:
    """Build one comparison CSV summary row."""
    best_row = training_run.best_row
    return {
        "rank": rank,
        "display_label": training_run.display_label,
        "run_id": training_run.run_id,
        "selection_metric": training_run.selection_metric,
        "best_epoch": best_row["epoch"],
        "best_step": best_row["step"],
        "selected_metric_value": best_row.get(training_run.selection_column, ""),
        "precision": best_row.get("val/precision", ""),
        "recall": best_row.get("val/recall", ""),
        "f1": best_row.get("val/F1", ""),
        "f2": best_row.get("val/F2", ""),
        "map50": best_row.get("val/mAP_50", ""),
        "map50_95": best_row.get("val/mAP_50_95", ""),
        "train_loss": best_row.get("train/loss", ""),
        "val_loss": best_row.get("val/loss", ""),
        "source_path": training_run.source_path.as_posix(),
    }


def _load_rfdetr_training_run(
    training_run_spec: Path | RfdetrTrainingRunSpec,
    selection_metric: str,
) -> RfdetrTrainingRun:
    """Load one RF-DETR training report for comparison."""
    if isinstance(training_run_spec, RfdetrTrainingRunSpec):
        report_path = training_run_spec.path
        display_label = training_run_spec.display_label
    else:
        report_path = training_run_spec
        display_label = None

    metrics_clean_path = _metrics_clean_path(report_path)
    if not metrics_clean_path.exists():
        raise ValueError(
            f"RF-DETR cleaned metrics CSV does not exist: {metrics_clean_path}"
        )
    metrics = read_rfdetr_clean_metrics(metrics_clean_path)
    best_row, normalized_metric, selection_column = select_best_rfdetr_training_row(
        metrics,
        selection_metric=selection_metric,
    )
    return RfdetrTrainingRun(
        run_id=report_path.name,
        display_label=display_label or report_path.name,
        source_path=report_path,
        metrics=metrics,
        selection_metric=normalized_metric,
        selection_column=selection_column,
        best_row=best_row,
    )


def _comparison_learning_rate_columns(
    training_runs: list[RfdetrTrainingRun],
) -> tuple[str, ...]:
    """Return sorted LR columns present in any compared run."""
    return tuple(
        sorted(
            {
                column_name
                for training_run in training_runs
                for column_name in lr_columns(training_run.metrics.columns)
            }
        )
    )


def parse_rfdetr_training_run_spec(value: str) -> RfdetrTrainingRunSpec:
    """Parse a training run comparison value as path or display-label=path."""
    if "=" not in value:
        return RfdetrTrainingRunSpec(path=Path(value))

    display_label, path_value = value.split("=", maxsplit=1)
    display_label = display_label.strip()
    if not display_label:
        raise ValueError("Training run display label cannot be empty")
    if not path_value:
        raise ValueError("Training run path cannot be empty")
    return RfdetrTrainingRunSpec(path=Path(path_value), display_label=display_label)


def load_rfdetr_training_runs(
    training_run_specs: list[Path | RfdetrTrainingRunSpec],
    selection_metric: str = "f2",
) -> list[RfdetrTrainingRun]:
    """Load RF-DETR training report directories for comparison."""
    if len(training_run_specs) < 2:
        raise ValueError("At least two --training-run values are required")
    return [
        _load_rfdetr_training_run(training_run_spec, selection_metric)
        for training_run_spec in training_run_specs
    ]


def write_rfdetr_training_comparison_csv(
    training_runs: list[RfdetrTrainingRun],
    summary_csv_path: Path,
) -> None:
    """Write selected RF-DETR training metrics for multiple runs."""
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", encoding="utf-8", newline="") as summary_file:
        fieldnames = [
            "rank",
            "display_label",
            "run_id",
            "selection_metric",
            "best_epoch",
            "best_step",
            "selected_metric_value",
            "precision",
            "recall",
            "f1",
            "f2",
            "map50",
            "map50_95",
            "train_loss",
            "val_loss",
            "source_path",
        ]
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, training_run in enumerate(
            _rank_training_runs(training_runs), start=1
        ):
            writer.writerow(_comparison_summary_row(rank, training_run))


def write_rfdetr_training_comparison_markdown(
    training_runs: list[RfdetrTrainingRun],
    summary_markdown_path: Path,
) -> None:
    """Write RF-DETR training comparison results as Markdown."""
    lines = [
        "# RF-DETR Training Run Comparison",
        "",
        "This comparison uses RF-DETR training and validation telemetry. It is "
        "not benchmark accuracy reporting.",
        "",
        "| Rank | Label | Best Epoch | Best Step | Metric | Value | Precision | "
        "Recall | F1 | F2 | mAP50 | mAP50-95 |",
        "| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |",
    ]
    for rank, training_run in enumerate(_rank_training_runs(training_runs), start=1):
        best_row = training_run.best_row
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    training_run.display_label,
                    str(int(best_row["epoch"])),
                    str(int(best_row["step"])),
                    training_run.selection_metric,
                    format_float(best_row.get(training_run.selection_column)),
                    format_float(best_row.get("val/precision")),
                    format_float(best_row.get("val/recall")),
                    format_float(best_row.get("val/F1")),
                    format_float(best_row.get("val/F2")),
                    format_float(best_row.get("val/mAP_50")),
                    format_float(best_row.get("val/mAP_50_95")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Use existing validation or benchmark commands for cleaned-dataset "
            "performance, and keep those artifacts under `runs/validation/` or "
            "`runs/benchmarks/`.",
        ]
    )
    summary_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rfdetr_training_comparison_reports(
    training_run_specs: list[Path | RfdetrTrainingRunSpec],
    output_dir: Path,
    selection_metric: str = "f2",
) -> dict[str, Path]:
    """Write comparison tables and overlay plots for RF-DETR training runs."""
    training_runs = load_rfdetr_training_runs(
        training_run_specs,
        selection_metric=selection_metric,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / "summary.csv"
    summary_markdown_path = output_dir / "summary.md"
    validation_f2_path = output_dir / "validation_f2_overlay.svg"
    map_path = output_dir / "map_overlay.svg"
    loss_path = output_dir / "loss_overlay.svg"
    learning_rate_path = output_dir / "learning_rate_overlay.svg"

    write_rfdetr_training_comparison_csv(training_runs, summary_csv_path)
    write_rfdetr_training_comparison_markdown(training_runs, summary_markdown_path)
    write_runs_overlay_chart(
        title="RF-DETR Validation F2 Comparison",
        y_label="F2",
        training_runs=training_runs,
        column_names=("val/F2",),
        output_path=validation_f2_path,
        y_domain=(0.0, 1.0),
    )
    write_runs_overlay_chart(
        title="RF-DETR Validation mAP Comparison",
        y_label="mAP",
        training_runs=training_runs,
        column_names=("val/mAP_50", "val/mAP_50_95"),
        output_path=map_path,
        y_domain=(0.0, 1.0),
    )
    write_runs_overlay_chart(
        title="RF-DETR Loss Comparison",
        y_label="Loss",
        training_runs=training_runs,
        column_names=("train/loss", "val/loss"),
        output_path=loss_path,
    )

    paths = {
        "summary_csv_path": summary_csv_path,
        "summary_markdown_path": summary_markdown_path,
        "validation_f2_path": validation_f2_path,
        "map_path": map_path,
        "loss_path": loss_path,
    }
    learning_rate_columns = _comparison_learning_rate_columns(training_runs)
    if learning_rate_columns:
        write_runs_overlay_chart(
            title="RF-DETR Learning Rate Comparison",
            y_label="Learning rate",
            training_runs=training_runs,
            column_names=learning_rate_columns,
            output_path=learning_rate_path,
        )
        paths["learning_rate_path"] = learning_rate_path
    return paths
