"""Tests for low-level detection metric helpers."""

from __future__ import annotations

import unittest

from face_detection_benchmark.evaluation import (
    PredictedBox,
    average_precision,
    iou_xyxy,
    match_predictions_at_threshold,
)


class EvaluationMetricsTest(unittest.TestCase):
    """Coverage for matching, IoU, and AP helpers."""

    def test_iou_xyxy(self) -> None:
        """Compute IoU for non-overlapping and partially overlapping boxes."""
        self.assertEqual(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(
            iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]),
            25 / 175,
        )

    def test_match_predictions_at_threshold_counts_unmatched_boxes(self) -> None:
        """Count false positives when a prediction cannot match ground truth."""
        ground_truth = {"image.jpg": [[0, 0, 10, 10]]}
        predictions = [
            PredictedBox("image.jpg", [0, 0, 10, 10], 0.9, "model"),
            PredictedBox("image.jpg", [30, 30, 40, 40], 0.8, "model"),
        ]

        counts = match_predictions_at_threshold(
            ground_truth_by_file_name=ground_truth,
            predictions=predictions,
            confidence_threshold=0.0,
            iou_threshold=0.5,
        )

        self.assertEqual(counts.true_positive_count, 1)
        self.assertEqual(counts.false_positive_count, 1)
        self.assertEqual(counts.false_negative_count, 0)

    def test_average_precision_is_one_for_perfect_ranked_predictions(self) -> None:
        """Compute perfect AP for perfectly ranked true-positive predictions."""
        ground_truth = {
            "first.jpg": [[0, 0, 10, 10]],
            "second.jpg": [[20, 20, 30, 30]],
        }
        predictions = [
            PredictedBox("first.jpg", [0, 0, 10, 10], 0.9, "model"),
            PredictedBox("second.jpg", [20, 20, 30, 30], 0.8, "model"),
        ]

        self.assertEqual(
            average_precision(
                ground_truth_by_file_name=ground_truth,
                predictions=predictions,
                iou_threshold=0.5,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
