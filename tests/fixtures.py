"""Shared test fixtures for benchmark evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

from face_detection_benchmark.evaluation import (
    DetectionMetrics,
    PredictedBox,
    match_predictions_at_threshold,
)


def write_coco_fixture(
    root: Path,
    prediction_file_name: str,
    detections: list[dict],
) -> tuple[Path, Path]:
    """Write a tiny COCO fixture and matching prediction JSONL."""
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "image.jpg").write_bytes(b"placeholder")
    (dataset_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "image.jpg",
                        "width": 100,
                        "height": 100,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "Human face"}],
            }
        ),
        encoding="utf-8",
    )
    predictions_path = root / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "file_name": prediction_file_name,
                "width": 100,
                "height": 100,
                "model_name": "test-model",
                "detections": detections,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset_dir, predictions_path


def evaluate_coco_predictions_for_test(
    ground_truth: dict[str, list[list[float]]],
    predictions: list[PredictedBox],
) -> DetectionMetrics:
    """Build a metrics object for report writer tests."""
    counts = match_predictions_at_threshold(
        ground_truth_by_file_name=ground_truth,
        predictions=predictions,
        confidence_threshold=0.0,
        iou_threshold=0.5,
    )
    return DetectionMetrics(
        model_name="model",
        image_count=1,
        ground_truth_count=1,
        prediction_count=len(predictions),
        confidence_threshold=0.0,
        iou_threshold=0.5,
        true_positive_count=counts.true_positive_count,
        false_positive_count=counts.false_positive_count,
        false_negative_count=counts.false_negative_count,
        precision=1.0,
        recall=1.0,
        f1=1.0,
        f2=1.0,
        ap50=1.0,
        ap75=1.0,
        map_50_95=1.0,
        ap_by_iou={"0.50": 1.0},
        confidence_sweep=[],
    )
