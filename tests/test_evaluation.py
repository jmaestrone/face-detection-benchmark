"""Tests for benchmark detection metric computation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from face_detection_benchmark.evaluation import (
    DEFAULT_SWEEP_THRESHOLDS,
    PredictedBox,
    average_precision,
    evaluate_coco_predictions,
    evaluate_confidence_thresholds,
    iou_xyxy,
    match_predictions_at_threshold,
)
from face_detection_benchmark.inference import rfdetr_model_name_from_weights
from face_detection_benchmark.reports import (
    append_results_row,
    write_results_leaderboard,
    write_threshold_validation_reports,
)


class EvaluationTest(unittest.TestCase):
    def test_rfdetr_model_name_from_weights(self) -> None:
        self.assertEqual(
            rfdetr_model_name_from_weights(Path("models/checkpoint_best_ema_2.pth")),
            "rfdetr-checkpoint-best-ema-2",
        )

    def test_iou_xyxy(self) -> None:
        self.assertEqual(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(
            iou_xyxy([0, 0, 10, 10], [5, 5, 15, 15]),
            25 / 175,
        )

    def test_match_predictions_at_threshold_counts_unmatched_boxes(self) -> None:
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

    def test_evaluate_coco_predictions_filters_to_target_category(self) -> None:
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
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir, predictions_path = write_coco_fixture(
                root=Path(temp_dir),
                prediction_file_name="image.jpg",
                detections=[],
            )

            metrics = evaluate_coco_predictions(dataset_dir, predictions_path)

            self.assertEqual(metrics.prediction_count, 0)
            self.assertEqual(metrics.false_negative_count, 1)

    def test_evaluate_confidence_thresholds_selects_best_threshold(self) -> None:
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
        self.assertNotIn(0.0, DEFAULT_SWEEP_THRESHOLDS)
        self.assertIn(0.005, DEFAULT_SWEEP_THRESHOLDS)
        self.assertIn(0.01, DEFAULT_SWEEP_THRESHOLDS)
        self.assertEqual(max(DEFAULT_SWEEP_THRESHOLDS), 0.8)

    def test_evaluate_confidence_thresholds_ties_choose_higher_threshold(self) -> None:
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

    def test_write_threshold_validation_reports_writes_tables_and_plots(self) -> None:
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
):
    """Build a metrics object for report writer tests."""
    from face_detection_benchmark.evaluation import DetectionMetrics

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


if __name__ == "__main__":
    unittest.main()
