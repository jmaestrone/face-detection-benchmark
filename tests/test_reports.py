"""Tests for generated report tables and charts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from face_detection_benchmark.evaluation import (
    PredictedBox,
    evaluate_confidence_thresholds,
)
from face_detection_benchmark.reports import (
    append_results_row,
    write_results_leaderboard,
    write_threshold_validation_reports,
)

from tests.fixtures import evaluate_coco_predictions_for_test, write_coco_fixture


class ReportWritersTest(unittest.TestCase):
    """Coverage for CSV, Markdown, JSON, and SVG report writers."""

    def test_write_threshold_validation_reports_writes_tables_and_plots(self) -> None:
        """Write validation report artifacts with expected table and chart content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir, predictions_path = write_coco_fixture(
                root=root,
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
                thresholds=(0.0, 0.5),
            )

            paths = write_threshold_validation_reports(
                result=result,
                output_dir=root / "validation",
                dataset_dir=dataset_dir,
                predictions_path=predictions_path,
            )

            self.assertTrue(paths["validation_path"].exists())
            self.assertTrue(paths["selected_threshold_path"].exists())
            self.assertIn(
                "confidence_threshold",
                paths["threshold_metrics_path"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "# Validation Threshold Metrics",
                paths["threshold_metrics_markdown_path"].read_text(encoding="utf-8"),
            )
            precision_recall_svg = paths["precision_recall_path"].read_text(
                encoding="utf-8",
            )
            f_scores_svg = paths["f_scores_path"].read_text(encoding="utf-8")
            self.assertIn("<svg", precision_recall_svg)
            self.assertIn("t=0.00-0.50", precision_recall_svg)
            self.assertIn("selected t=0.50", precision_recall_svg)
            self.assertIn(">0.95</text>", precision_recall_svg)
            self.assertIn("<svg", f_scores_svg)
            self.assertIn("threshold=0.00", f_scores_svg)
            self.assertIn("threshold=0.50", f_scores_svg)
            self.assertIn("selected t=0.50", f_scores_svg)
            self.assertIn(">0.5</text>", f_scores_svg)

    def test_append_results_row_writes_header_once(self) -> None:
        """Append cumulative result rows without duplicating the CSV header."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = Path(temp_dir) / "results.csv"
            ground_truth = {"image.jpg": [[0, 0, 10, 10]]}
            predictions = [PredictedBox("image.jpg", [0, 0, 10, 10], 0.9, "model")]
            metrics = evaluate_coco_predictions_for_test(
                ground_truth=ground_truth,
                predictions=predictions,
            )

            append_results_row(
                metrics=metrics,
                results_table_path=results_path,
                run_id="run-a",
                dataset_dir=Path("dataset"),
                predictions_path=Path("predictions.jsonl"),
            )
            append_results_row(
                metrics=metrics,
                results_table_path=results_path,
                run_id="run-b",
                dataset_dir=Path("dataset"),
                predictions_path=Path("predictions.jsonl"),
            )

            lines = results_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("run_id,model_name", lines[0])
            self.assertIn("run-a,model", lines[1])
            self.assertIn("run-b,model", lines[2])

    def test_write_results_leaderboard_writes_markdown_table(self) -> None:
        """Write a Markdown leaderboard from the cumulative results CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_path = Path(temp_dir) / "results.csv"
            leaderboard_path = Path(temp_dir) / "results.md"
            metrics = evaluate_coco_predictions_for_test(
                ground_truth={"image.jpg": [[0, 0, 10, 10]]},
                predictions=[PredictedBox("image.jpg", [0, 0, 10, 10], 0.9, "model")],
            )
            append_results_row(
                metrics=metrics,
                results_table_path=results_path,
                run_id="run-a",
                dataset_dir=Path("dataset"),
                predictions_path=Path("predictions.jsonl"),
            )

            write_results_leaderboard(results_path, leaderboard_path)

            markdown = leaderboard_path.read_text(encoding="utf-8")
            self.assertIn("# Face Detection Benchmark Results", markdown)
            self.assertIn("| 1 | run-a | model |", markdown)
            self.assertIn("1.0000", markdown)


if __name__ == "__main__":
    unittest.main()
