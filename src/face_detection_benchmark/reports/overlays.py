"""OpenCV overlay rendering for classified prediction matches."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from face_detection_benchmark.config import FACE_CATEGORY_NAME
from face_detection_benchmark.datasets import CocoDetectionDataset
from face_detection_benchmark.evaluation import (
    classify_detection_matches,
    load_evaluation_inputs,
)
from face_detection_benchmark.evaluation.metrics import fbeta_score, precision, recall
from face_detection_benchmark.evaluation.types import (
    DetectionCounts,
    DetectionMatchClassification,
    GroundTruthBox,
    PredictedBox,
)

GREEN_BGR = (0, 255, 0)
YELLOW_BGR = (0, 255, 255)
RED_BGR = (0, 0, 255)
HEADER_BGR = (32, 32, 32)
WHITE_BGR = (255, 255, 255)
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class PredictionOverlaySpec:
    """One labeled prediction file and confidence threshold for overlay rendering."""

    label: str
    predictions_path: Path
    confidence_threshold: float


@dataclass(frozen=True)
class PredictionOverlayResult:
    """Output artifact paths and summary rows for one overlay render run."""

    output_dir: Path
    model_dirs: dict[str, Path]
    comparison_dir: Path | None
    summary_csv_path: Path
    summary_json_path: Path
    summary_rows: list[dict[str, str | int | float]]
    image_count: int


def parse_prediction_overlay_spec(raw_spec: str) -> PredictionOverlaySpec:
    """Parse a CLI prediction spec in label=path.jsonl:threshold format."""
    label, separator, path_and_threshold = raw_spec.partition("=")
    if not separator or not label:
        raise ValueError(
            "--prediction-spec must use label=path/to/predictions.jsonl:threshold"
        )
    if not LABEL_PATTERN.fullmatch(label):
        raise ValueError(
            "Prediction spec labels may contain only letters, numbers, dots, "
            "underscores, and hyphens, and must start with a letter or number"
        )

    path_text, threshold_separator, threshold_text = path_and_threshold.rpartition(":")
    if not threshold_separator or not path_text or not threshold_text:
        raise ValueError(
            "--prediction-spec must use label=path/to/predictions.jsonl:threshold"
        )
    try:
        confidence_threshold = float(threshold_text)
    except ValueError as error:
        raise ValueError("Prediction spec threshold must be a number") from error
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("Prediction spec threshold must be between 0.0 and 1.0")

    return PredictionOverlaySpec(
        label=label,
        predictions_path=Path(path_text),
        confidence_threshold=confidence_threshold,
    )


def render_prediction_overlays(
    *,
    dataset_dir: Path,
    prediction_specs: list[PredictionOverlaySpec],
    output_dir: Path,
    category_name: str = FACE_CATEGORY_NAME,
    iou_threshold: float = 0.5,
) -> PredictionOverlayResult:
    """Render classified prediction overlays and summary artifacts."""
    if not prediction_specs:
        raise ValueError("At least one --prediction-spec is required")

    seen_labels: set[str] = set()
    for prediction_spec in prediction_specs:
        if prediction_spec.label in seen_labels:
            raise ValueError(
                f"Duplicate prediction spec label: {prediction_spec.label}"
            )
        seen_labels.add(prediction_spec.label)

    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths_by_label: dict[str, dict[str, Path]] = {}
    model_dirs: dict[str, Path] = {}
    summary_rows: list[dict[str, str | int | float]] = []
    reference_dataset: CocoDetectionDataset | None = None

    for prediction_spec in prediction_specs:
        inputs = load_evaluation_inputs(
            dataset_dir=dataset_dir,
            predictions_path=prediction_spec.predictions_path,
            category_name=category_name,
        )
        reference_dataset = reference_dataset or inputs.dataset
        classification = classify_detection_matches(
            ground_truth_by_file_name=inputs.ground_truth_by_file_name,
            predictions=inputs.predictions,
            confidence_threshold=prediction_spec.confidence_threshold,
            iou_threshold=iou_threshold,
        )
        counts = _counts_from_classification(classification)
        summary_rows.append(
            _summary_row(
                label=prediction_spec.label,
                model_name=inputs.model_name,
                predictions_path=prediction_spec.predictions_path,
                confidence_threshold=prediction_spec.confidence_threshold,
                iou_threshold=iou_threshold,
                image_count=len(inputs.dataset.images),
                ground_truth_count=sum(
                    len(boxes) for boxes in inputs.ground_truth_by_file_name.values()
                ),
                prediction_count=len(inputs.predictions),
                counts=counts,
            )
        )

        model_dir = output_dir / "models" / prediction_spec.label
        model_dirs[prediction_spec.label] = model_dir
        rendered_paths_by_label[prediction_spec.label] = _write_model_overlays(
            dataset=inputs.dataset,
            classification=classification,
            output_dir=model_dir,
        )

    comparison_dir = None
    if len(prediction_specs) > 1 and reference_dataset is not None:
        comparison_dir = output_dir / "comparison"
        _write_comparison_overlays(
            dataset=reference_dataset,
            prediction_specs=prediction_specs,
            rendered_paths_by_label=rendered_paths_by_label,
            output_dir=comparison_dir,
        )

    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"
    _write_summary_csv(summary_rows, summary_csv_path)
    _write_summary_json(
        summary_rows=summary_rows,
        summary_json_path=summary_json_path,
        dataset_dir=dataset_dir,
        category_name=category_name,
        iou_threshold=iou_threshold,
    )
    return PredictionOverlayResult(
        output_dir=output_dir,
        model_dirs=model_dirs,
        comparison_dir=comparison_dir,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
        summary_rows=summary_rows,
        image_count=len(reference_dataset.images) if reference_dataset else 0,
    )


def _write_model_overlays(
    *,
    dataset: CocoDetectionDataset,
    classification: DetectionMatchClassification,
    output_dir: Path,
) -> dict[str, Path]:
    true_positive_predictions = _predictions_by_file(
        classification.true_positive_predictions
    )
    false_positive_predictions = _predictions_by_file(
        classification.false_positive_predictions
    )
    false_negative_ground_truths = _ground_truths_by_file(
        classification.false_negative_ground_truths
    )

    rendered_paths: dict[str, Path] = {}
    for image_record in dataset.images:
        image_bgr = _read_image(image_record.image_path)
        annotated_image = image_bgr.copy()
        for ground_truth in false_negative_ground_truths.get(
            image_record.file_name, []
        ):
            _draw_labeled_box(annotated_image, ground_truth.bbox_xyxy, "FN", RED_BGR)
        for prediction in true_positive_predictions.get(image_record.file_name, []):
            _draw_labeled_box(
                annotated_image,
                prediction.bbox_xyxy,
                f"TP {prediction.confidence:.2f}",
                GREEN_BGR,
            )
        for prediction in false_positive_predictions.get(image_record.file_name, []):
            _draw_labeled_box(
                annotated_image,
                prediction.bbox_xyxy,
                f"FP {prediction.confidence:.2f}",
                YELLOW_BGR,
            )

        output_path = output_dir / image_record.file_name
        _write_image(output_path, annotated_image)
        rendered_paths[image_record.file_name] = output_path
    return rendered_paths


def _write_comparison_overlays(
    *,
    dataset: CocoDetectionDataset,
    prediction_specs: list[PredictionOverlaySpec],
    rendered_paths_by_label: dict[str, dict[str, Path]],
    output_dir: Path,
) -> None:
    for image_record in dataset.images:
        panels = []
        for prediction_spec in prediction_specs:
            rendered_path = rendered_paths_by_label[prediction_spec.label][
                image_record.file_name
            ]
            panel = _read_image(rendered_path)
            panels.append(
                _add_comparison_header(
                    panel,
                    (
                        f"{prediction_spec.label} "
                        f"t={prediction_spec.confidence_threshold:.2f}"
                    ),
                )
            )
        comparison_image = _horizontal_concat(panels)
        _write_image(output_dir / image_record.file_name, comparison_image)


def _draw_labeled_box(
    image_bgr: Any,
    bbox_xyxy: list[float],
    label: str,
    color_bgr: tuple[int, int, int],
) -> None:
    import cv2

    x_min, y_min, x_max, y_max = [round(value) for value in bbox_xyxy]
    cv2.rectangle(image_bgr, (x_min, y_min), (x_max, y_max), color_bgr, 2)
    cv2.putText(
        image_bgr,
        label,
        (x_min, max(16, y_min - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_bgr,
        1,
        cv2.LINE_AA,
    )


def _add_comparison_header(image_bgr: Any, label: str) -> Any:
    import cv2

    header_height = 28
    panel = cv2.copyMakeBorder(
        image_bgr,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=HEADER_BGR,
    )
    cv2.putText(
        panel,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE_BGR,
        1,
        cv2.LINE_AA,
    )
    return panel


def _horizontal_concat(images_bgr: list[Any]) -> Any:
    import cv2

    if not images_bgr:
        raise ValueError("At least one image is required for comparison overlays")
    return cv2.hconcat(images_bgr)


def _read_image(image_path: Path) -> Any:
    import cv2

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image_bgr


def _write_image(output_path: Path, image_bgr: Any) -> None:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise ValueError(f"Could not write overlay image: {output_path}")


def _predictions_by_file(
    predictions: list[PredictedBox],
) -> dict[str, list[PredictedBox]]:
    predictions_by_file: dict[str, list[PredictedBox]] = {}
    for prediction in predictions:
        predictions_by_file.setdefault(prediction.file_name, []).append(prediction)
    return predictions_by_file


def _ground_truths_by_file(
    ground_truths: list[GroundTruthBox],
) -> dict[str, list[GroundTruthBox]]:
    ground_truths_by_file: dict[str, list[GroundTruthBox]] = {}
    for ground_truth in ground_truths:
        ground_truths_by_file.setdefault(ground_truth.file_name, []).append(
            ground_truth
        )
    return ground_truths_by_file


def _counts_from_classification(
    classification: DetectionMatchClassification,
) -> DetectionCounts:
    return DetectionCounts(
        true_positive_count=len(classification.true_positive_predictions),
        false_positive_count=len(classification.false_positive_predictions),
        false_negative_count=len(classification.false_negative_ground_truths),
    )


def _summary_row(
    *,
    label: str,
    model_name: str,
    predictions_path: Path,
    confidence_threshold: float,
    iou_threshold: float,
    image_count: int,
    ground_truth_count: int,
    prediction_count: int,
    counts: DetectionCounts,
) -> dict[str, str | int | float]:
    precision_value = precision(
        counts.true_positive_count,
        counts.false_positive_count,
    )
    recall_value = recall(
        counts.true_positive_count,
        counts.false_negative_count,
    )
    return {
        "label": label,
        "model_name": model_name,
        "prediction_path": predictions_path.as_posix(),
        "threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "image_count": image_count,
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "true_positive_count": counts.true_positive_count,
        "false_positive_count": counts.false_positive_count,
        "false_negative_count": counts.false_negative_count,
        "precision": precision_value,
        "recall": recall_value,
        "f1": fbeta_score(precision_value, recall_value, beta=1.0),
        "f2": fbeta_score(precision_value, recall_value, beta=2.0),
    }


def _write_summary_csv(
    summary_rows: list[dict[str, str | int | float]],
    summary_csv_path: Path,
) -> None:
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "model_name",
        "prediction_path",
        "threshold",
        "iou_threshold",
        "image_count",
        "ground_truth_count",
        "prediction_count",
        "true_positive_count",
        "false_positive_count",
        "false_negative_count",
        "precision",
        "recall",
        "f1",
        "f2",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def _write_summary_json(
    *,
    summary_rows: list[dict[str, str | int | float]],
    summary_json_path: Path,
    dataset_dir: Path,
    category_name: str,
    iou_threshold: float,
) -> None:
    payload = {
        "dataset_dir": dataset_dir.as_posix(),
        "category_name": category_name,
        "iou_threshold": iou_threshold,
        "models": summary_rows,
    }
    summary_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
