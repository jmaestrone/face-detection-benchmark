"""Low-level matching and metric helpers for detection evaluation."""

from __future__ import annotations

from typing import Iterable

from face_detection_benchmark.evaluation.types import DetectionCounts, PredictedBox


def _best_unmatched_ground_truth_index(
    prediction: PredictedBox,
    ground_truth_boxes: list[list[float]],
    matched_indexes: set[int],
    iou_threshold: float,
) -> int | None:
    """Find the highest-IoU unmatched ground-truth box for one prediction."""
    best_index = None
    best_iou = 0.0
    for ground_truth_index, ground_truth_box in enumerate(ground_truth_boxes):
        if ground_truth_index in matched_indexes:
            continue
        overlap = iou_xyxy(prediction.bbox_xyxy, ground_truth_box)
        if overlap >= iou_threshold and overlap > best_iou:
            best_index = ground_truth_index
            best_iou = overlap
    return best_index


def precision(true_positive_count: int, false_positive_count: int) -> float:
    """Compute precision from true-positive and false-positive counts."""
    denominator = true_positive_count + false_positive_count
    if denominator == 0:
        return 0.0
    return round(true_positive_count / denominator, 6)


def recall(true_positive_count: int, false_negative_count: int) -> float:
    """Compute recall from true-positive and false-negative counts."""
    denominator = true_positive_count + false_negative_count
    if denominator == 0:
        return 0.0
    return round(true_positive_count / denominator, 6)


def f1_score(precision_value: float, recall_value: float) -> float:
    """Compute F1 from precision and recall."""
    return fbeta_score(precision_value, recall_value, beta=1.0)


def fbeta_score(precision_value: float, recall_value: float, beta: float) -> float:
    """Compute F-beta from precision, recall, and beta."""
    beta_squared = beta * beta
    numerator = (1 + beta_squared) * precision_value * recall_value
    denominator = (beta_squared * precision_value) + recall_value
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def mean_metric(values: Iterable[float]) -> float:
    """Compute a rounded arithmetic mean for metric values."""
    values_list = list(values)
    if not values_list:
        return 0.0
    return round(sum(values_list) / len(values_list), 6)


def sweep_row(
    threshold: float,
    counts: DetectionCounts,
) -> dict[str, float | int]:
    """Build one confidence-threshold sweep row from detection counts."""
    precision_value = precision(
        counts.true_positive_count,
        counts.false_positive_count,
    )
    recall_value = recall(
        counts.true_positive_count,
        counts.false_negative_count,
    )
    return {
        "confidence_threshold": threshold,
        "true_positive_count": counts.true_positive_count,
        "false_positive_count": counts.false_positive_count,
        "false_negative_count": counts.false_negative_count,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_score(precision_value, recall_value),
        "f2": fbeta_score(precision_value, recall_value, beta=2.0),
    }


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
        predictions,
        key=lambda prediction: prediction.confidence,
        reverse=True,
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
        precisions.append(precision(cumulative_tp, cumulative_fp))
        recalls.append(cumulative_tp / total_ground_truth)

    interpolated_precisions = []
    for recall_threshold in (index / 100 for index in range(101)):
        precision_values = [
            precision_value
            for precision_value, recall_value in zip(precisions, recalls)
            if recall_value >= recall_threshold
        ]
        interpolated_precisions.append(max(precision_values, default=0.0))
    return round(mean_metric(interpolated_precisions), 6)


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
    x_min = max(first_box[0], second_box[0])
    y_min = max(first_box[1], second_box[1])
    x_max = min(first_box[2], second_box[2])
    y_max = min(first_box[3], second_box[3])
    intersection_width = max(0.0, x_max - x_min)
    intersection_height = max(0.0, y_max - y_min)
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
