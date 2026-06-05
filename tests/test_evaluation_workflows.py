"""Tests for COCO prediction evaluation workflows."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from face_detection_benchmark.evaluation import evaluate_coco_predictions

from tests.fixtures import write_coco_fixture


class EvaluationWorkflowTest(unittest.TestCase):
    """Coverage for dataset loading and full prediction evaluation."""

    def test_evaluate_coco_predictions_filters_to_target_category(self) -> None:
        """Ignore non-target categories when evaluating predictions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
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
                            },
                            {
                                "id": 2,
                                "image_id": 1,
                                "category_id": 0,
                                "bbox": [50, 50, 10, 10],
                                "area": 100,
                                "iscrowd": 0,
                            },
                        ],
                        "categories": [
                            {"id": 0, "name": "ignored"},
                            {"id": 1, "name": "Human face"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            predictions_path = root / "predictions.jsonl"
            predictions_path.write_text(
                json.dumps(
                    {
                        "file_name": "image.jpg",
                        "width": 100,
                        "height": 100,
                        "model_name": "test-model",
                        "detections": [
                            {
                                "bbox_xyxy": [0, 0, 10, 10],
                                "bbox_xywh": [0, 0, 10, 10],
                                "confidence": 0.9,
                                "class_id": 1,
                                "class_name": "Human face",
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            metrics = evaluate_coco_predictions(
                dataset_dir=dataset_dir,
                predictions_path=predictions_path,
            )

            self.assertEqual(metrics.ground_truth_count, 1)
            self.assertEqual(metrics.prediction_count, 1)
            self.assertEqual(metrics.true_positive_count, 1)
            self.assertEqual(metrics.ap50, 1.0)
            self.assertEqual(metrics.map_50_95, 1.0)
            self.assertEqual(metrics.f2, 1.0)

    def test_evaluate_coco_predictions_rejects_mismatched_filenames(self) -> None:
        """Reject prediction files with no filenames matching the dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir, predictions_path = write_coco_fixture(
                root=Path(temp_dir),
                prediction_file_name="different.jpg",
                detections=[
                    {
                        "bbox_xyxy": [0, 0, 10, 10],
                        "bbox_xywh": [0, 0, 10, 10],
                        "confidence": 0.9,
                        "class_id": 1,
                        "class_name": "Human face",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "no image filenames matching"):
                evaluate_coco_predictions(dataset_dir, predictions_path)

    def test_evaluate_coco_predictions_allows_matching_rows_with_no_detections(
        self,
    ) -> None:
        """Allow matching prediction rows that contain no detections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir, predictions_path = write_coco_fixture(
                root=Path(temp_dir),
                prediction_file_name="image.jpg",
                detections=[],
            )

            metrics = evaluate_coco_predictions(dataset_dir, predictions_path)

            self.assertEqual(metrics.prediction_count, 0)
            self.assertEqual(metrics.false_negative_count, 1)


if __name__ == "__main__":
    unittest.main()
