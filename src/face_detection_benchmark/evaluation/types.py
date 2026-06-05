"""Shared data structures for detection evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from face_detection_benchmark.datasets import CocoDetectionDataset


@dataclass(frozen=True)
class PredictedBox:
    """One predicted bounding box used for metric computation."""

    file_name: str
    bbox_xyxy: list[float]
    confidence: float
    model_name: str


@dataclass(frozen=True)
class PredictionRows:
    """Prediction row metadata and boxes loaded from JSONL."""

    file_names: set[str]
    boxes: list[PredictedBox]


@dataclass(frozen=True)
class DetectionCounts:
    """Confusion counts for one confidence and IoU threshold."""

    true_positive_count: int
    false_positive_count: int
    false_negative_count: int


@dataclass(frozen=True)
class DetectionMetrics:
    """Summary detection metrics for one prediction file."""

    model_name: str
    image_count: int
    ground_truth_count: int
    prediction_count: int
    confidence_threshold: float
    iou_threshold: float
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    precision: float
    recall: float
    f1: float
    f2: float
    ap50: float
    ap75: float
    map_50_95: float
    ap_by_iou: dict[str, float]
    confidence_sweep: list[dict[str, float | int]]


@dataclass(frozen=True)
class ThresholdValidationResult:
    """Threshold sweep metrics and selected operating point for validation."""

    model_name: str
    image_count: int
    ground_truth_count: int
    prediction_count: int
    iou_threshold: float
    selection_metric: str
    selected_threshold: float
    selected_metrics: dict[str, float | int]
    threshold_metrics: list[dict[str, float | int]]


@dataclass(frozen=True)
class EvaluationInputs:
    """Shared normalized inputs for detection evaluation helpers."""

    dataset: CocoDetectionDataset
    ground_truth_by_file_name: dict[str, list[list[float]]]
    predictions: list[PredictedBox]
    model_name: str
