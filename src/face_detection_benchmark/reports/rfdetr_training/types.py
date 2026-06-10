"""Shared types and constants for RF-DETR training reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SELECTION_METRIC_COLUMNS = {
    "f2": "val/F2",
    "f1": "val/F1",
    "precision": "val/precision",
    "recall": "val/recall",
    "map50": "val/mAP_50",
    "map50-95": "val/mAP_50_95",
}
BASE_METRIC_COLUMNS = (
    "train/loss",
    "val/loss",
    "val/precision",
    "val/recall",
    "val/F1",
    "val/F2",
    "val/mAP_50",
    "val/mAP_50_95",
    "val/mAP_75",
)
MAP_COLUMNS = (
    "val/mAP_50",
    "val/mAP_50_95",
    "val/mAP_75",
    "val/ema_mAP_50",
    "val/ema_mAP_50_95",
)
SCORE_COLUMNS = ("val/precision", "val/recall", "val/F1", "val/F2")
LOSS_COLUMNS = ("train/loss", "val/loss")


@dataclass(frozen=True)
class RfdetrTrainingMetrics:
    """Parsed and merged RF-DETR training metrics."""

    rows: list[dict[str, float | int]]
    columns: list[str]
    source_row_count: int
    merged_row_count: int


@dataclass(frozen=True)
class RfdetrTrainingReport:
    """RF-DETR training report data and selected best row."""

    run_id: str
    source_metrics_path: Path
    metrics: RfdetrTrainingMetrics
    selection_metric: str
    selection_column: str
    best_row: dict[str, float | int]


@dataclass(frozen=True)
class RfdetrTrainingRunSpec:
    """A training report path with an optional display label."""

    path: Path
    display_label: str | None = None


@dataclass(frozen=True)
class RfdetrTrainingRun:
    """One loaded RF-DETR training report for comparison."""

    run_id: str
    display_label: str
    source_path: Path
    metrics: RfdetrTrainingMetrics
    selection_metric: str
    selection_column: str
    best_row: dict[str, float | int]
