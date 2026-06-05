"""Tests for generated report tables and charts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from face_detection_benchmark.evaluation import (
    PredictedBox,
    evaluate_confidence_thresholds,
    threshold_validation_to_json_dict,
)
from face_detection_benchmark.evaluation.types import ThresholdValidationResult
from face_detection_benchmark.reports import (
    append_results_row,
    write_results_leaderboard,
    write_threshold_validation_reports,
    write_validation_comparison_reports,
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

    def test_write_validation_comparison_reports_writes_overlay_outputs(self) -> None:
        """Write comparison tables and overlay charts for validation runs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_run = root / "first-run"
            second_run = root / "second-run"
            first_run.mkdir()
            second_run.mkdir()
            first_result = threshold_validation_result_for_test(
                model_name="model-a",
                selected_threshold=0.3,
                precision=0.8,
                recall=0.8,
                f1=0.8,
                f2=0.8,
            )
            second_result = threshold_validation_result_for_test(
                model_name="model-b",
                selected_threshold=0.2,
                precision=0.9,
                recall=0.75,
                f1=0.82,
                f2=0.78,
            )
            (first_run / "threshold_validation.json").write_text(
                json_dumps(threshold_validation_to_json_dict(first_result)),
                encoding="utf-8",
            )
            (second_run / "threshold_validation.json").write_text(
                json_dumps(threshold_validation_to_json_dict(second_result)),
                encoding="utf-8",
            )

            paths = write_validation_comparison_reports(
                validation_run_paths=[first_run, second_run],
                output_dir=root / "comparison",
            )

            self.assertIn(
                "model-a",
                paths["summary_markdown_path"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "model-b",
                paths["summary_csv_path"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "Precision vs Recall Comparison",
                paths["precision_recall_path"].read_text(encoding="utf-8"),
            )
            self.assertIn(
                "F Scores by Threshold Comparison",
                paths["f_scores_path"].read_text(encoding="utf-8"),
            )


def threshold_validation_result_for_test(
    model_name: str,
    selected_threshold: float,
    precision: float,
    recall: float,
    f1: float,
    f2: float,
) -> ThresholdValidationResult:
    """Build a validation result for report tests."""
    selected_metrics = {
        "confidence_threshold": selected_threshold,
        "true_positive_count": 8,
        "false_positive_count": 2,
        "false_negative_count": 2,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }
    return ThresholdValidationResult(
        model_name=model_name,
        image_count=10,
        ground_truth_count=10,
        prediction_count=12,
        iou_threshold=0.5,
        selection_metric="f2",
        selected_threshold=selected_threshold,
        selected_metrics=selected_metrics,
        threshold_metrics=[
            {
                "confidence_threshold": 0.1,
                "true_positive_count": 7,
                "false_positive_count": 3,
                "false_negative_count": 3,
                "precision": 0.7,
                "recall": 0.7,
                "f1": 0.7,
                "f2": 0.7,
            },
            selected_metrics,
        ],
    )


def json_dumps(payload: dict) -> str:
    """Serialize compact JSON for test fixture files."""
    return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
