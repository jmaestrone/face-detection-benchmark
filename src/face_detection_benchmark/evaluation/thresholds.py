"""Threshold validation helpers for detection evaluation."""

from __future__ import annotations

from typing import Iterable

from face_detection_benchmark.evaluation.metrics import (
    match_predictions_at_threshold,
    sweep_row,
)
from face_detection_benchmark.evaluation.types import (
    EvaluationInputs,
    ThresholdValidationResult,
)


def _validate_selection_metric(selection_metric: str) -> None:
    """Validate the metric used to choose a confidence threshold."""
    if selection_metric not in {"precision", "recall", "f1", "f2"}:
        raise ValueError(
            "--selection-metric must be one of precision, recall, f1, or f2"
        )


def _threshold_values(thresholds: Iterable[float]) -> list[float]:
    """Validate and normalize confidence threshold values."""
    threshold_values = [float(threshold) for threshold in thresholds]
    if not threshold_values:
        raise ValueError("At least one threshold is required")
    for threshold in threshold_values:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold values must be between 0 and 1")
    return threshold_values


def _select_threshold_metrics(
    threshold_metrics: list[dict[str, float | int]],
    selection_metric: str,
) -> dict[str, float | int]:
    """Choose the best threshold row, breaking ties toward higher thresholds."""
    return max(
        threshold_metrics,
        key=lambda threshold_row: (
            float(threshold_row[selection_metric]),
            float(threshold_row["confidence_threshold"]),
        ),
    )


def evaluate_confidence_thresholds_from_inputs(
    inputs: EvaluationInputs,
    iou_threshold: float,
    thresholds: Iterable[float],
    selection_metric: str,
) -> ThresholdValidationResult:
    """Evaluate threshold rows from already-loaded evaluation inputs."""
    _validate_selection_metric(selection_metric)
    threshold_values = _threshold_values(thresholds)
    threshold_metrics = [
        sweep_row(
            threshold=threshold,
            counts=match_predictions_at_threshold(
                ground_truth_by_file_name=inputs.ground_truth_by_file_name,
                predictions=inputs.predictions,
                confidence_threshold=threshold,
                iou_threshold=iou_threshold,
            ),
        )
        for threshold in threshold_values
    ]
    selected_metrics = _select_threshold_metrics(
        threshold_metrics=threshold_metrics,
        selection_metric=selection_metric,
    )

    return ThresholdValidationResult(
        model_name=inputs.model_name,
        image_count=len(inputs.dataset.images),
        ground_truth_count=sum(
            len(boxes) for boxes in inputs.ground_truth_by_file_name.values()
        ),
        prediction_count=len(inputs.predictions),
        iou_threshold=iou_threshold,
        selection_metric=selection_metric,
        selected_threshold=float(selected_metrics["confidence_threshold"]),
        selected_metrics=selected_metrics,
        threshold_metrics=threshold_metrics,
    )
