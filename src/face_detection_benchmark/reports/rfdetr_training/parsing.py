"""Parsing and selection helpers for RF-DETR training metrics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from face_detection_benchmark.reports.rfdetr_training.types import (
    BASE_METRIC_COLUMNS,
    LOSS_COLUMNS,
    MAP_COLUMNS,
    SELECTION_METRIC_COLUMNS,
    RfdetrTrainingMetrics,
)


def _parse_float(value: Any) -> float | None:
    """Parse a finite float, returning None for blank or invalid values."""
    if value is None or value == "":
        return None
    try:
        parsed_value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed_value) or math.isinf(parsed_value):
        return None
    return parsed_value


def _parse_int(value: Any) -> int | None:
    """Parse an integer-like CSV value."""
    parsed_value = _parse_float(value)
    if parsed_value is None:
        return None
    return int(parsed_value)


def _is_useful_column(column_name: str | None) -> bool:
    """Return whether a raw RF-DETR metric column belongs in reports."""
    if not column_name:
        return False
    return (
        column_name in BASE_METRIC_COLUMNS
        or column_name in MAP_COLUMNS
        or column_name.startswith("train/lr")
        or column_name.startswith("val/ema_mAP")
    )


def _ordered_metric_columns(observed_columns: set[str]) -> list[str]:
    """Order cleaned metric columns for stable CSV output."""
    ordered_columns = [
        column_name
        for column_name in (
            *LOSS_COLUMNS,
            "val/precision",
            "val/recall",
            "val/F1",
            "val/F2",
            *MAP_COLUMNS,
        )
        if column_name in observed_columns
    ]
    ordered_columns.extend(
        sorted(
            column_name
            for column_name in observed_columns
            if column_name.startswith("val/ema_mAP")
            and column_name not in ordered_columns
        )
    )
    ordered_columns.extend(
        sorted(
            column_name
            for column_name in observed_columns
            if column_name.startswith("train/lr")
        )
    )
    return ordered_columns


def _with_computed_f2(
    metrics_row: dict[str, float | int],
) -> dict[str, float | int]:
    """Add computed validation F2 when precision and recall are present."""
    if "val/precision" not in metrics_row or "val/recall" not in metrics_row:
        return metrics_row
    precision = float(metrics_row["val/precision"])
    recall = float(metrics_row["val/recall"])
    denominator = (4.0 * precision) + recall
    metrics_row["val/F2"] = (
        0.0 if denominator == 0 else (5.0 * precision * recall) / denominator
    )
    return metrics_row


def _normalize_selection_metric(selection_metric: str) -> str:
    """Normalize metric aliases used by CLI and tests."""
    normalized_metric = selection_metric.strip().lower().replace("_", "-")
    aliases = {
        "map-50": "map50",
        "map_50": "map50",
        "map": "map50-95",
        "map-50-95": "map50-95",
        "map_50_95": "map50-95",
    }
    normalized_metric = aliases.get(normalized_metric, normalized_metric)
    if normalized_metric not in SELECTION_METRIC_COLUMNS:
        valid_metrics = ", ".join(SELECTION_METRIC_COLUMNS)
        raise ValueError(
            f"Unsupported selection metric '{selection_metric}'. "
            f"Expected one of: {valid_metrics}"
        )
    return normalized_metric


def _read_clean_metrics_csv(metrics_clean_path: Path) -> RfdetrTrainingMetrics:
    """Load a previously written merged RF-DETR metrics CSV."""
    with metrics_clean_path.open("r", encoding="utf-8", newline="") as metrics_file:
        reader = csv.DictReader(metrics_file)
        cleaned_rows = []
        for source_row in reader:
            metrics_row: dict[str, float | int] = {}
            for column_name, raw_value in source_row.items():
                if column_name in {"epoch", "step"}:
                    parsed_value = _parse_int(raw_value)
                else:
                    parsed_value = _parse_float(raw_value)
                if parsed_value is not None:
                    metrics_row[column_name] = parsed_value
            if "epoch" in metrics_row and "step" in metrics_row:
                cleaned_rows.append(metrics_row)
        columns = list(reader.fieldnames or [])
    if not cleaned_rows:
        raise ValueError(f"No parseable RF-DETR metrics rows in {metrics_clean_path}")
    return RfdetrTrainingMetrics(
        rows=cleaned_rows,
        columns=columns,
        source_row_count=len(cleaned_rows),
        merged_row_count=len(cleaned_rows),
    )


def lr_columns(columns: list[str]) -> list[str]:
    """Return learning-rate columns from a cleaned RF-DETR metrics schema."""
    return [
        column_name for column_name in columns if column_name.startswith("train/lr")
    ]


def has_metric_column(
    metrics_rows: list[dict[str, float | int]],
    column_name: str,
) -> bool:
    """Return whether any metrics row contains a column."""
    return any(column_name in metrics_row for metrics_row in metrics_rows)


def csv_value(value: Any) -> Any:
    """Format None as an empty CSV cell while preserving numeric values."""
    if value is None:
        return ""
    return value


def format_float(value: Any) -> str:
    """Format a table float with four decimal places."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def parse_rfdetr_training_metrics(metrics_csv_path: Path) -> RfdetrTrainingMetrics:
    """Parse sparse RF-DETR metrics rows into merged numeric rows."""
    if not metrics_csv_path.exists():
        raise ValueError(f"RF-DETR metrics CSV does not exist: {metrics_csv_path}")

    with metrics_csv_path.open("r", encoding="utf-8", newline="") as metrics_file:
        source_rows = list(csv.DictReader(metrics_file))

    merged_rows_by_key: dict[tuple[int, int], dict[str, float | int]] = {}
    observed_columns: set[str] = set()
    for source_row in source_rows:
        epoch = _parse_int(source_row.get("epoch"))
        step = _parse_int(source_row.get("step"))
        if epoch is None or step is None:
            continue

        metrics_row = merged_rows_by_key.setdefault(
            (epoch, step),
            {"epoch": epoch, "step": step},
        )
        for column_name, raw_value in source_row.items():
            if column_name in {"epoch", "step"} or not _is_useful_column(column_name):
                continue
            parsed_value = _parse_float(raw_value)
            if parsed_value is None:
                continue
            metrics_row[column_name] = parsed_value
            observed_columns.add(column_name)

    merged_rows = [
        _with_computed_f2(metrics_row)
        for metrics_row in sorted(
            merged_rows_by_key.values(),
            key=lambda metrics_row: (
                int(metrics_row["epoch"]),
                int(metrics_row["step"]),
            ),
        )
    ]
    observed_columns.update(
        "val/F2" for metrics_row in merged_rows if "val/F2" in metrics_row
    )
    if not merged_rows:
        raise ValueError(f"No parseable RF-DETR metrics rows in {metrics_csv_path}")

    columns = ["epoch", "step"] + _ordered_metric_columns(observed_columns)
    return RfdetrTrainingMetrics(
        rows=merged_rows,
        columns=columns,
        source_row_count=len(source_rows),
        merged_row_count=len(merged_rows),
    )


def select_best_rfdetr_training_row(
    metrics: RfdetrTrainingMetrics,
    selection_metric: str = "f2",
) -> tuple[dict[str, float | int], str, str]:
    """Select the best RF-DETR validation row for the requested metric."""
    normalized_metric = _normalize_selection_metric(selection_metric)
    selection_column = SELECTION_METRIC_COLUMNS[normalized_metric]
    candidate_rows = [
        metrics_row for metrics_row in metrics.rows if selection_column in metrics_row
    ]
    if not candidate_rows:
        raise ValueError(
            f"Selection metric '{selection_metric}' is unavailable; "
            f"expected column {selection_column}"
        )
    best_row = max(
        candidate_rows,
        key=lambda metrics_row: (
            float(metrics_row[selection_column]),
            int(metrics_row["epoch"]),
            int(metrics_row["step"]),
        ),
    )
    return best_row, normalized_metric, selection_column


def read_rfdetr_clean_metrics(metrics_clean_path: Path) -> RfdetrTrainingMetrics:
    """Read cleaned RF-DETR metrics from a training report directory."""
    return _read_clean_metrics_csv(metrics_clean_path)
