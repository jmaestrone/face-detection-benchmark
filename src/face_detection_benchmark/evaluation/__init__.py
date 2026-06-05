"""Detection evaluation orchestration for benchmark predictions."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from face_detection_benchmark.config import FACE_CATEGORY_NAME
from face_detection_benchmark.datasets import (
    CocoDetectionDataset,
    load_coco_detection_dataset,
)
from face_detection_benchmark.evaluation.metrics import (
    average_precision,
    classify_detection_matches,
    f1_score,
    fbeta_score,
    iou_xyxy,
    match_predictions_at_threshold,
    mean_metric,
    precision,
    recall,
    sweep_row,
)
from face_detection_benchmark.evaluation.thresholds import (
    evaluate_confidence_thresholds_from_inputs,
)
from face_detection_benchmark.evaluation.types import (
    DetectionMatchClassification,
    DetectionMetrics,
    EvaluationInputs,
    GroundTruthBox,
    PredictedBox,
    PredictionRows,
    ThresholdValidationResult,
)

DEFAULT_IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
DEFAULT_SWEEP_THRESHOLDS = (
    0.005,
    0.01,
    *(round(index * 0.05, 2) for index in range(1, 17)),
)

__all__ = [
    "DEFAULT_IOU_THRESHOLDS",
    "DEFAULT_SWEEP_THRESHOLDS",
    "DetectionMetrics",
    "DetectionMatchClassification",
    "GroundTruthBox",
    "PredictedBox",
    "PredictionRows",
    "ThresholdValidationResult",
    "average_precision",
    "classify_detection_matches",
    "evaluate_coco_predictions",
    "evaluate_confidence_thresholds",
    "iou_xyxy",
    "match_predictions_at_threshold",
    "metrics_to_json_dict",
    "read_prediction_boxes",
    "read_prediction_rows",
    "threshold_validation_to_json_dict",
]


def _ground_truth_boxes_by_file_name(
    dataset: CocoDetectionDataset,
    category_ids: set[int],
) -> dict[str, list[list[float]]]:
    """Group target-category ground-truth boxes by image filename."""
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


def _model_name_for_predictions(
    predictions: list[PredictedBox],
    predictions_path: Path,
) -> str:
    """Resolve one model name for a prediction file."""
    if not predictions:
        return predictions_path.stem
    model_names = sorted({prediction.model_name for prediction in predictions})
    if len(model_names) == 1:
        return model_names[0]
    return "mixed"


def _xywh_to_xyxy(bbox_xywh: list[float]) -> list[float]:
    """Convert a COCO xywh bounding box to xyxy."""
    x_min, y_min, width, height = bbox_xywh
    return [x_min, y_min, x_min + width, y_min + height]


def _load_evaluation_inputs(
    dataset_dir: Path,
    predictions_path: Path,
    category_name: str,
) -> EvaluationInputs:
    """Load and align COCO ground truth with normalized prediction rows."""
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


def metrics_to_json_dict(metrics: DetectionMetrics) -> dict[str, Any]:
    """Convert detection metrics to a JSON-serializable dictionary."""
    return asdict(metrics)


def threshold_validation_to_json_dict(
    result: ThresholdValidationResult,
) -> dict[str, Any]:
    """Convert threshold validation output to a JSON-serializable dictionary."""
    return asdict(result)


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
    precision_value = precision(
        counts.true_positive_count,
        counts.false_positive_count,
    )
    recall_value = recall(
        counts.true_positive_count,
        counts.false_negative_count,
    )

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
            sweep_row(
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
        precision=precision_value,
        recall=recall_value,
        f1=f1_score(precision_value, recall_value),
        f2=fbeta_score(precision_value, recall_value, beta=2.0),
        ap50=ap_by_iou.get("0.50", 0.0),
        ap75=ap_by_iou.get("0.75", 0.0),
        map_50_95=mean_metric(ap_by_iou.values()),
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
    inputs = _load_evaluation_inputs(
        dataset_dir=dataset_dir,
        predictions_path=predictions_path,
        category_name=category_name,
    )
    return evaluate_confidence_thresholds_from_inputs(
        inputs=inputs,
        iou_threshold=iou_threshold,
        thresholds=thresholds,
        selection_metric=selection_metric,
    )
