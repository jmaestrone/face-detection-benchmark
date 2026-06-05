"""Tests for confidence-threshold validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from face_detection_benchmark.evaluation import (
    DEFAULT_SWEEP_THRESHOLDS,
    evaluate_confidence_thresholds,
)

from tests.fixtures import write_coco_fixture


class ThresholdValidationTest(unittest.TestCase):
    """Coverage for threshold sweeps and selected operating points."""

    def test_evaluate_confidence_thresholds_selects_best_threshold(self) -> None:
        """Select the threshold with the best requested validation metric."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir, predictions_path = write_coco_fixture(
                root=Path(temp_dir),
                prediction_file_name="image.jpg",
                detections=[
                    {
                        "bbox_xyxy": [0, 0, 10, 10],
                        "bbox_xywh": [0, 0, 10, 10],
                        "confidence": 0.9,
                        "class_id": 1,
                        "class_name": "Human face",
                    },
                    {
                        "bbox_xyxy": [50, 50, 60, 60],
                        "bbox_xywh": [50, 50, 10, 10],
                        "confidence": 0.3,
                        "class_id": 1,
                        "class_name": "Human face",
                    },
                ],
            )

            result = evaluate_confidence_thresholds(
                dataset_dir=dataset_dir,
                predictions_path=predictions_path,
                thresholds=(0.0, 0.5),
                selection_metric="f2",
            )

            self.assertEqual(result.selected_threshold, 0.5)
            self.assertEqual(result.selected_metrics["precision"], 1.0)
            self.assertEqual(result.selected_metrics["recall"], 1.0)

    def test_evaluate_confidence_thresholds_uses_nonzero_default_thresholds(
        self,
    ) -> None:
        """Keep the default validation sweep away from zero threshold."""
        self.assertNotIn(0.0, DEFAULT_SWEEP_THRESHOLDS)
        self.assertIn(0.005, DEFAULT_SWEEP_THRESHOLDS)
        self.assertIn(0.01, DEFAULT_SWEEP_THRESHOLDS)
        self.assertEqual(max(DEFAULT_SWEEP_THRESHOLDS), 0.8)

    def test_evaluate_confidence_thresholds_ties_choose_higher_threshold(self) -> None:
        """Break equal metric scores by choosing the higher confidence threshold."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir, predictions_path = write_coco_fixture(
                root=Path(temp_dir),
                prediction_file_name="image.jpg",
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

            result = evaluate_confidence_thresholds(
                dataset_dir=dataset_dir,
                predictions_path=predictions_path,
                thresholds=(0.005, 0.01, 0.5),
                selection_metric="f2",
            )

            self.assertEqual(result.selected_threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
