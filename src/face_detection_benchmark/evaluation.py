"""Detection metric computation for benchmark predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from face_detection_benchmark.config import FACE_CATEGORY_NAME
from face_detection_benchmark.datasets import (
    CocoDetectionDataset,
    load_coco_detection_dataset,
)

DEFAULT_IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
DEFAULT_SWEEP_THRESHOLDS = (
    0.005,
    0.01,
    *(round(index * 0.05, 2) for index in range(1, 17)),
)


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


def evaluate_coco_predictions(
    dataset_dir: Path,
    predictions_path: Path,
    category_name: str = FACE_CATEGORY_NAME,
    confidence_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    iou_thresholds: Iterable[float] = DEFAULT_IOU_THRESHOLDS,
    sweep_thresholds: Iterable[float] | None = None,
) -> DetectionMetrics:
    """Evaluate normalized prediction JSONL against a COCO detection dataset."""
    inputs = _load_evaluation_inputs(
        dataset_dir=dataset_dir,
        predictions_path=predictions_path,
        category_name=category_name,
    )

    counts = match_predictions_at_threshold(
        ground_truth_by_file_name=inputs.ground_truth_by_file_name,
        predictions=inputs.predictions,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )
    precision = _precision(counts.true_positive_count, counts.false_positive_count)
    recall = _recall(counts.true_positive_count, counts.false_negative_count)

    ap_by_iou = {
        f"{threshold:.2f}": average_precision(
            ground_truth_by_file_name=inputs.ground_truth_by_file_name,
            predictions=inputs.predictions,
            iou_threshold=threshold,
        )
        for threshold in iou_thresholds
    }
    confidence_sweep = []
    if sweep_thresholds is not None:
        confidence_sweep = [
            _sweep_row(
                threshold=threshold,
                counts=match_predictions_at_threshold(
                    ground_truth_by_file_name=inputs.ground_truth_by_file_name,
                    predictions=inputs.predictions,
                    confidence_threshold=threshold,
                    iou_threshold=iou_threshold,
                ),
            )
            for threshold in sweep_thresholds
        ]

    return DetectionMetrics(
        model_name=inputs.model_name,
        image_count=len(inputs.dataset.images),
        ground_truth_count=sum(
            len(boxes) for boxes in inputs.ground_truth_by_file_name.values()
        ),
        prediction_count=len(inputs.predictions),
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        true_positive_count=counts.true_positive_count,
        false_positive_count=counts.false_positive_count,
        false_negative_count=counts.false_negative_count,
        precision=precision,
        recall=recall,
        f1=_f1(precision, recall),
        f2=_fbeta(precision, recall, beta=2.0),
        ap50=ap_by_iou.get("0.50", 0.0),
        ap75=ap_by_iou.get("0.75", 0.0),
        map_50_95=_mean(ap_by_iou.values()),
        ap_by_iou=ap_by_iou,
        confidence_sweep=confidence_sweep,
    )


def evaluate_confidence_thresholds(
    dataset_dir: Path,
    predictions_path: Path,
    category_name: str = FACE_CATEGORY_NAME,
    iou_threshold: float = 0.5,
    thresholds: Iterable[float] = DEFAULT_SWEEP_THRESHOLDS,
    selection_metric: str = "f2",
) -> ThresholdValidationResult:
    """Evaluate many confidence thresholds and select one validation threshold."""
    if selection_metric not in {"precision", "recall", "f1", "f2"}:
        raise ValueError(
            "--selection-metric must be one of precision, recall, f1, or f2"
        )

    threshold_values = [float(threshold) for threshold in thresholds]
    if not threshold_values:
        raise ValueError("At least one threshold is required")
    for threshold in threshold_values:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold values must be between 0 and 1")

    inputs = _load_evaluation_inputs(
        dataset_dir=dataset_dir,
        predictions_path=predictions_path,
        category_name=category_name,
    )
    threshold_metrics = [
        _sweep_row(
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


def read_prediction_rows(predictions_path: Path) -> PredictionRows:
    """Read normalized prediction JSONL row filenames and boxes."""
    file_names: set[str] = set()
    predictions: list[PredictedBox] = []
    with predictions_path.open("r", encoding="utf-8") as predictions_file:
        for line in predictions_file:
            if not line.strip():
                continue
            row = json.loads(line)
            file_name = str(row["file_name"])
            file_names.add(file_name)
            model_name = str(row.get("model_name") or predictions_path.stem)
            for detection in row.get("detections", []):
                confidence = detection.get("confidence")
                if confidence is None:
                    confidence = 1.0
                predictions.append(
                    PredictedBox(
                        file_name=file_name,
                        bbox_xyxy=[float(value) for value in detection["bbox_xyxy"]],
                        confidence=float(confidence),
                        model_name=model_name,
                    )
                )
    return PredictionRows(file_names=file_names, boxes=predictions)


def read_prediction_boxes(predictions_path: Path) -> list[PredictedBox]:
    """Read normalized prediction JSONL into metric-ready boxes."""
    return read_prediction_rows(predictions_path).boxes


def average_precision(
    ground_truth_by_file_name: dict[str, list[list[float]]],
    predictions: list[PredictedBox],
    iou_threshold: float,
) -> float:
    """Compute 101-point interpolated AP at one IoU threshold."""
    total_ground_truth = sum(len(boxes) for boxes in ground_truth_by_file_name.values())
    if total_ground_truth == 0:
        return 0.0

    sorted_predictions = sorted(
        predictions, key=lambda prediction: prediction.confidence, reverse=True
    )
    matched_indexes: dict[str, set[int]] = {
        file_name: set() for file_name in ground_truth_by_file_name
    }
    true_positives: list[int] = []
    false_positives: list[int] = []

    for prediction in sorted_predictions:
        match_index = _best_unmatched_ground_truth_index(
            prediction=prediction,
            ground_truth_boxes=ground_truth_by_file_name.get(prediction.file_name, []),
            matched_indexes=matched_indexes.setdefault(prediction.file_name, set()),
            iou_threshold=iou_threshold,
        )
        if match_index is None:
            true_positives.append(0)
            false_positives.append(1)
        else:
            matched_indexes[prediction.file_name].add(match_index)
            true_positives.append(1)
            false_positives.append(0)

    if not true_positives:
        return 0.0

    cumulative_tp = 0
    cumulative_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for true_positive, false_positive in zip(true_positives, false_positives):
        cumulative_tp += true_positive
        cumulative_fp += false_positive
        precisions.append(_precision(cumulative_tp, cumulative_fp))
        recalls.append(cumulative_tp / total_ground_truth)

    interpolated_precisions = []
    for recall_threshold in (index / 100 for index in range(101)):
        precision_values = [
            precision
            for precision, recall in zip(precisions, recalls)
            if recall >= recall_threshold
        ]
        interpolated_precisions.append(max(precision_values, default=0.0))
    return round(_mean(interpolated_precisions), 6)


def match_predictions_at_threshold(
    ground_truth_by_file_name: dict[str, list[list[float]]],
    predictions: list[PredictedBox],
    confidence_threshold: float,
    iou_threshold: float,
) -> DetectionCounts:
    """Greedily match predictions to ground truth at fixed thresholds."""
    matched_indexes: dict[str, set[int]] = {
        file_name: set() for file_name in ground_truth_by_file_name
    }
    true_positive_count = 0
    false_positive_count = 0
    filtered_predictions = sorted(
        (
            prediction
            for prediction in predictions
            if prediction.confidence >= confidence_threshold
        ),
        key=lambda prediction: prediction.confidence,
        reverse=True,
    )

    for prediction in filtered_predictions:
        match_index = _best_unmatched_ground_truth_index(
            prediction=prediction,
            ground_truth_boxes=ground_truth_by_file_name.get(prediction.file_name, []),
            matched_indexes=matched_indexes.setdefault(prediction.file_name, set()),
            iou_threshold=iou_threshold,
        )
        if match_index is None:
            false_positive_count += 1
        else:
            matched_indexes[prediction.file_name].add(match_index)
            true_positive_count += 1

    matched_count = sum(len(indexes) for indexes in matched_indexes.values())
    ground_truth_count = sum(len(boxes) for boxes in ground_truth_by_file_name.values())
    return DetectionCounts(
        true_positive_count=true_positive_count,
        false_positive_count=false_positive_count,
        false_negative_count=ground_truth_count - matched_count,
    )


def iou_xyxy(first_box: list[float], second_box: list[float]) -> float:
    """Compute IoU for two xyxy bounding boxes."""
    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])
    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection_area = intersection_width * intersection_height
    first_area = max(0.0, first_box[2] - first_box[0]) * max(
        0.0, first_box[3] - first_box[1]
    )
    second_area = max(0.0, second_box[2] - second_box[0]) * max(
        0.0, second_box[3] - second_box[1]
    )
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def metrics_to_json_dict(metrics: DetectionMetrics) -> dict[str, Any]:
    """Convert detection metrics to a JSON-serializable dictionary."""
    return asdict(metrics)


def threshold_validation_to_json_dict(
    result: ThresholdValidationResult,
) -> dict[str, Any]:
    """Convert threshold validation output to a JSON-serializable dictionary."""
    return asdict(result)


def _load_evaluation_inputs(
    dataset_dir: Path,
    predictions_path: Path,
    category_name: str,
) -> EvaluationInputs:
    if not predictions_path.exists():
        raise ValueError(f"Predictions file does not exist: {predictions_path}")

    dataset = load_coco_detection_dataset(dataset_dir)
    category_ids = {
        int(category["id"])
        for category in dataset.categories
        if str(category.get("name")) == category_name
    }
    if not category_ids:
        raise ValueError(f"Category {category_name!r} not found in {dataset_dir}")

    ground_truth_by_file_name = _ground_truth_boxes_by_file_name(dataset, category_ids)
    prediction_rows = read_prediction_rows(predictions_path)
    image_file_names = {image.file_name for image in dataset.images}
    overlapping_file_names = prediction_rows.file_names & image_file_names
    if prediction_rows.file_names and not overlapping_file_names:
        raise ValueError(
            "Prediction file has no image filenames matching the COCO dataset. "
            "Run predictions on this benchmark split or remap filenames before "
            "evaluation."
        )
    relevant_predictions = [
        prediction
        for prediction in prediction_rows.boxes
        if prediction.file_name in image_file_names
    ]
    return EvaluationInputs(
        dataset=dataset,
        ground_truth_by_file_name=ground_truth_by_file_name,
        predictions=relevant_predictions,
        model_name=_model_name_for_predictions(relevant_predictions, predictions_path),
    )


def _ground_truth_boxes_by_file_name(
    dataset: CocoDetectionDataset,
    category_ids: set[int],
) -> dict[str, list[list[float]]]:
    image_by_id = {image.id: image for image in dataset.images}
    boxes_by_file_name = {image.file_name: [] for image in dataset.images}
    for annotation in dataset.annotations:
        if annotation.category_id not in category_ids:
            continue
        image = image_by_id.get(annotation.image_id)
        if image is None:
            continue
        boxes_by_file_name[image.file_name].append(_xywh_to_xyxy(annotation.bbox_xywh))
    return boxes_by_file_name


def _best_unmatched_ground_truth_index(
    prediction: PredictedBox,
    ground_truth_boxes: list[list[float]],
    matched_indexes: set[int],
    iou_threshold: float,
) -> int | None:
    best_index = None
    best_iou = 0.0
    for index, ground_truth_box in enumerate(ground_truth_boxes):
        if index in matched_indexes:
            continue
        overlap = iou_xyxy(prediction.bbox_xyxy, ground_truth_box)
        if overlap >= iou_threshold and overlap > best_iou:
            best_index = index
            best_iou = overlap
    return best_index


def _sweep_row(
    threshold: float,
    counts: DetectionCounts,
) -> dict[str, float | int]:
    precision = _precision(counts.true_positive_count, counts.false_positive_count)
    recall = _recall(counts.true_positive_count, counts.false_negative_count)
    return {
        "confidence_threshold": threshold,
        "true_positive_count": counts.true_positive_count,
        "false_positive_count": counts.false_positive_count,
        "false_negative_count": counts.false_negative_count,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "f2": _fbeta(precision, recall, beta=2.0),
    }


def _select_threshold_metrics(
    threshold_metrics: list[dict[str, float | int]],
    selection_metric: str,
) -> dict[str, float | int]:
    return max(
        threshold_metrics,
        key=lambda row: (
            float(row[selection_metric]),
            float(row["confidence_threshold"]),
        ),
    )


def _model_name_for_predictions(
    predictions: list[PredictedBox],
    predictions_path: Path,
) -> str:
    if not predictions:
        return predictions_path.stem
    model_names = sorted({prediction.model_name for prediction in predictions})
    if len(model_names) == 1:
        return model_names[0]
    return "mixed"


def _xywh_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    x, y, width, height = bbox_xywh
    return [x, y, x + width, y + height]


def _precision(true_positive_count: int, false_positive_count: int) -> float:
    denominator = true_positive_count + false_positive_count
    if denominator == 0:
        return 0.0
    return round(true_positive_count / denominator, 6)


def _recall(true_positive_count: int, false_negative_count: int) -> float:
    denominator = true_positive_count + false_negative_count
    if denominator == 0:
        return 0.0
    return round(true_positive_count / denominator, 6)


def _f1(precision: float, recall: float) -> float:
    return _fbeta(precision, recall, beta=1.0)


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta_squared = beta * beta
    numerator = (1 + beta_squared) * precision * recall
    denominator = (beta_squared * precision) + recall
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return round(sum(values_list) / len(values_list), 6)
