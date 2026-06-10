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
    ValidationRunSpec,
    append_results_row,
    load_rfdetr_training_runs,
    load_validation_runs,
    parse_rfdetr_training_metrics,
    parse_rfdetr_training_run_spec,
    parse_validation_run_spec,
    select_best_rfdetr_training_row,
    summarize_video_predictions,
    write_results_leaderboard,
    write_rfdetr_training_comparison_reports,
    write_rfdetr_training_report,
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

    def test_parse_validation_run_spec_accepts_plain_path(self) -> None:
        """Keep the old unlabeled --validation-run path form supported."""
        validation_run_spec = parse_validation_run_spec("runs/validation/run-a")

        self.assertEqual(validation_run_spec.path, Path("runs/validation/run-a"))
        self.assertIsNone(validation_run_spec.display_label)

    def test_parse_validation_run_spec_accepts_display_label(self) -> None:
        """Parse comparison display labels from label=path values."""
        validation_run_spec = parse_validation_run_spec(
            "RF-DETR=runs/validation/rfdetr-validation",
        )

        self.assertEqual(
            validation_run_spec.path,
            Path("runs/validation/rfdetr-validation"),
        )
        self.assertEqual(validation_run_spec.display_label, "RF-DETR")

    def test_load_validation_runs_falls_back_to_model_name_label(self) -> None:
        """Use model_name as the display label for old path inputs."""
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

            validation_runs = load_validation_runs([first_run, second_run])

            self.assertEqual(
                [validation_run.display_label for validation_run in validation_runs],
                ["model-a", "model-b"],
            )

    def test_summarize_video_predictions_writes_per_video_outputs(self) -> None:
        """Summarize extracted-frame predictions without video accuracy metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "metadata.jsonl"
            predictions_path = root / "predictions.jsonl"
            output_dir = root / "video-summaries"
            write_jsonl(
                metadata_path,
                [
                    {
                        "file_name": "video-a_frame000000_0000000000ms.jpg",
                        "output_path": "video-a_frame000000_0000000000ms.jpg",
                        "source_video": "videos/video-a.mp4",
                        "video_stem": "video-a",
                        "frame_index": 0,
                        "timestamp_seconds": 0.0,
                        "width": 640,
                        "height": 480,
                        "source_fps": 30.0,
                    },
                    {
                        "file_name": "video-a_frame000030_0000001000ms.jpg",
                        "output_path": "video-a_frame000030_0000001000ms.jpg",
                        "source_video": "videos/video-a.mp4",
                        "video_stem": "video-a",
                        "frame_index": 30,
                        "timestamp_seconds": 1.0,
                        "width": 640,
                        "height": 480,
                        "source_fps": 30.0,
                    },
                    {
                        "file_name": "video-b_frame000000_0000000000ms.jpg",
                        "output_path": "video-b_frame000000_0000000000ms.jpg",
                        "source_video": "videos/video-b.mp4",
                        "video_stem": "video-b",
                        "frame_index": 0,
                        "timestamp_seconds": 0.0,
                        "width": 1280,
                        "height": 720,
                    },
                ],
            )
            write_jsonl(
                predictions_path,
                [
                    {
                        "file_name": "video-a_frame000000_0000000000ms.jpg",
                        "image_path": "frames/video-a_frame000000_0000000000ms.jpg",
                        "width": 640,
                        "height": 480,
                        "model_name": "rfdetr-test",
                        "detections": [
                            {"confidence": 0.9, "bbox_xyxy": [1, 2, 3, 4]},
                            {"confidence": 0.1, "bbox_xyxy": [5, 6, 7, 8]},
                        ],
                        "timing_ms": {"inference": 10.0},
                    },
                    {
                        "file_name": "video-a_frame000030_0000001000ms.jpg",
                        "image_path": "frames/video-a_frame000030_0000001000ms.jpg",
                        "width": 640,
                        "height": 480,
                        "model_name": "rfdetr-test",
                        "detections": [],
                        "timing_ms": {"inference": 20.0},
                    },
                    {
                        "file_name": "video-b_frame000000_0000000000ms.jpg",
                        "image_path": "frames/video-b_frame000000_0000000000ms.jpg",
                        "width": 1280,
                        "height": 720,
                        "model_name": "rfdetr-test",
                        "detections": [{"confidence": 0.8, "bbox_xyxy": [1, 1, 2, 2]}],
                    },
                ],
            )

            report_paths = summarize_video_predictions(
                predictions_path=predictions_path,
                metadata_path=metadata_path,
                output_dir=output_dir,
                confidence_threshold=0.25,
            )

            aggregate = json.loads(
                report_paths["summary_json_path"].read_text(encoding="utf-8")
            )
            video_a = json.loads(
                (output_dir / "video-a" / "summary.json").read_text(encoding="utf-8")
            )
            video_b = json.loads(
                (output_dir / "video-b" / "summary.json").read_text(encoding="utf-8")
            )
            csv_text = report_paths["summary_csv_path"].read_text(encoding="utf-8")

            self.assertEqual(aggregate["video_count"], 2)
            self.assertEqual(aggregate["processed_frame_count"], 3)
            self.assertEqual(aggregate["total_detections"], 2)
            self.assertNotIn("precision", aggregate)
            self.assertNotIn("recall", aggregate)
            self.assertNotIn("map_50_95", aggregate)
            self.assertEqual(video_a["source_video"], "videos/video-a.mp4")
            self.assertEqual(video_a["video_stem"], "video-a")
            self.assertEqual(video_a["processed_frame_count"], 2)
            self.assertEqual(video_a["frames_with_faces"], 1)
            self.assertEqual(video_a["total_detections"], 1)
            self.assertEqual(video_a["timestamps_with_faces"], [0.0])
            self.assertEqual(video_a["sampled_fps"], 1.0)
            self.assertEqual(video_a["source_fps"], 30.0)
            self.assertEqual(video_a["latency"]["total_ms"], 30.0)
            self.assertEqual(video_a["latency"]["processed_fps"], 66.6667)
            self.assertEqual(video_b["frames_with_faces"], 1)
            self.assertIsNone(video_b["sampled_fps"])
            self.assertIn("video-a", csv_text)
            self.assertIn("processed_fps", csv_text)

    def test_write_validation_comparison_reports_uses_display_labels(self) -> None:
        """Write labeled comparison outputs without long model names in plot labels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_run = root / "first-run"
            second_run = root / "second-run"
            first_run.mkdir()
            second_run.mkdir()
            first_result = threshold_validation_result_for_test(
                model_name="rfdetr-14-11-29",
                selected_threshold=0.3,
                precision=0.8,
                recall=0.8,
                f1=0.8,
                f2=0.8,
            )
            second_result = threshold_validation_result_for_test(
                model_name="insightface-buffalo-l-det1280-thr005",
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
                validation_run_paths=[
                    ValidationRunSpec(path=first_run, display_label="RF-DETR"),
                    ValidationRunSpec(path=second_run, display_label="InsightFace"),
                ],
                output_dir=root / "comparison",
            )

            summary_csv = paths["summary_csv_path"].read_text(encoding="utf-8")
            summary_markdown = paths["summary_markdown_path"].read_text(
                encoding="utf-8",
            )
            precision_recall_svg = paths["precision_recall_path"].read_text(
                encoding="utf-8",
            )
            f_scores_svg = paths["f_scores_path"].read_text(encoding="utf-8")
            self.assertIn("display_label,run_id,model_name", summary_csv)
            self.assertIn("RF-DETR", summary_csv)
            self.assertIn("InsightFace", summary_csv)
            self.assertIn("| 1 | RF-DETR |", summary_markdown)
            self.assertIn("| 2 | InsightFace |", summary_markdown)
            self.assertIn(">RF-DETR</text>", precision_recall_svg)
            self.assertIn(">InsightFace</text>", precision_recall_svg)
            self.assertIn(">RF-DETR F1</text>", f_scores_svg)
            self.assertIn(">RF-DETR F2</text>", f_scores_svg)
            self.assertNotIn(
                "insightface-buffalo-l-det1280-thr005",
                precision_recall_svg,
            )

    def test_precision_recall_plot_falls_back_for_lower_metric_models(self) -> None:
        """Show multiple lower-domain points when the primary cutoff is too strict."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = threshold_validation_result_for_test(
                model_name="lower-metric-model",
                selected_threshold=0.35,
                precision=0.575,
                recall=0.267442,
                f1=0.365079,
                f2=0.299479,
                threshold_metrics=[
                    threshold_row_for_test(0.005, 0.000051, 0.802326),
                    threshold_row_for_test(0.2, 0.245283, 0.302326),
                    threshold_row_for_test(0.35, 0.575, 0.267442),
                    threshold_row_for_test(0.5, 0.782609, 0.209302),
                    threshold_row_for_test(0.8, 0.0, 0.0),
                ],
            )

            paths = write_threshold_validation_reports(
                result=result,
                output_dir=root / "validation",
                dataset_dir=root / "dataset",
                predictions_path=root / "predictions.jsonl",
            )

            precision_recall_svg = paths["precision_recall_path"].read_text(
                encoding="utf-8",
            )
            self.assertIn("t=0.20", precision_recall_svg)
            self.assertIn("selected t=0.35", precision_recall_svg)
            self.assertIn("t=0.50", precision_recall_svg)
            self.assertNotIn("t=0.005", precision_recall_svg)

    def test_precision_recall_overlay_falls_back_per_model(self) -> None:
        """Keep comparison overlays informative for models below the primary cutoff."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            strong_run = root / "strong-run"
            lower_run = root / "lower-run"
            strong_run.mkdir()
            lower_run.mkdir()
            strong_result = threshold_validation_result_for_test(
                model_name="strong-model",
                selected_threshold=0.3,
                precision=0.8,
                recall=0.8,
                f1=0.8,
                f2=0.8,
            )
            lower_result = threshold_validation_result_for_test(
                model_name="lower-metric-model",
                selected_threshold=0.35,
                precision=0.575,
                recall=0.267442,
                f1=0.365079,
                f2=0.299479,
                threshold_metrics=[
                    threshold_row_for_test(0.2, 0.245283, 0.302326),
                    threshold_row_for_test(0.35, 0.575, 0.267442),
                    threshold_row_for_test(0.5, 0.782609, 0.209302),
                ],
            )
            (strong_run / "threshold_validation.json").write_text(
                json_dumps(threshold_validation_to_json_dict(strong_result)),
                encoding="utf-8",
            )
            (lower_run / "threshold_validation.json").write_text(
                json_dumps(threshold_validation_to_json_dict(lower_result)),
                encoding="utf-8",
            )

            paths = write_validation_comparison_reports(
                validation_run_paths=[strong_run, lower_run],
                output_dir=root / "comparison",
            )

            precision_recall_svg = paths["precision_recall_path"].read_text(
                encoding="utf-8",
            )
            self.assertIn("lower-metric-model", precision_recall_svg)
            self.assertIn("threshold=0.20", precision_recall_svg)
            self.assertIn("lower-metric-model t=0.35", precision_recall_svg)

    def test_parse_rfdetr_training_metrics_merges_sparse_rows(self) -> None:
        """Merge sparse RF-DETR train, validation, and LR rows by epoch/step."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.csv"
            write_rfdetr_metrics_fixture(metrics_path)

            metrics = parse_rfdetr_training_metrics(metrics_path)

            self.assertEqual(metrics.source_row_count, 5)
            self.assertEqual(metrics.merged_row_count, 3)
            self.assertIn("val/F2", metrics.columns)
            first_validation_row = metrics.rows[1]
            self.assertEqual(first_validation_row["epoch"], 0)
            self.assertEqual(first_validation_row["step"], 20)
            self.assertEqual(first_validation_row["train/loss"], 6.0)
            self.assertEqual(first_validation_row["val/loss"], 4.0)
            self.assertEqual(first_validation_row["train/lr"], 0.001)
            self.assertNotIn("val/loss", metrics.rows[0])
            self.assertAlmostEqual(first_validation_row["val/F2"], 0.7758620690)

    def test_select_best_rfdetr_training_row_supports_metrics_and_ties(self) -> None:
        """Select best RF-DETR row by F2, F1, mAP, and later-step tie breaks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.csv"
            write_rfdetr_metrics_fixture(metrics_path)
            metrics = parse_rfdetr_training_metrics(metrics_path)

            best_f2, metric_name, column = select_best_rfdetr_training_row(metrics)
            best_f1, _, _ = select_best_rfdetr_training_row(metrics, "f1")
            best_map, _, _ = select_best_rfdetr_training_row(metrics, "map50-95")

            self.assertEqual(metric_name, "f2")
            self.assertEqual(column, "val/F2")
            self.assertEqual(best_f2["step"], 30)
            self.assertEqual(best_f1["step"], 30)
            self.assertEqual(best_map["step"], 30)

    def test_write_rfdetr_training_report_writes_files(self) -> None:
        """Write single-run RF-DETR training reports and SVG charts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / "metrics.csv"
            write_rfdetr_metrics_fixture(metrics_path)

            paths = write_rfdetr_training_report(
                metrics_csv_path=metrics_path,
                output_dir=root / "report",
                run_id="run-a",
            )

            self.assertTrue(paths["metrics_clean_path"].exists())
            self.assertTrue(paths["metrics_markdown_path"].exists())
            summary = paths["summary_path"].read_text(encoding="utf-8")
            metrics_markdown = paths["metrics_markdown_path"].read_text(
                encoding="utf-8"
            )
            self.assertIn("not benchmark accuracy reporting", summary)
            self.assertIn("data/benchmark/target-video-test-3fps-clean/test", summary)
            self.assertIn("# RF-DETR Training Metrics", metrics_markdown)
            self.assertIn("`val/F2`", metrics_markdown)
            self.assertIn(
                "val/F2",
                paths["metrics_clean_path"].read_text(encoding="utf-8"),
            )
            self.assertIn("<svg", paths["loss_path"].read_text(encoding="utf-8"))
            self.assertIn(
                "RF-DETR Validation Precision",
                paths["score_path"].read_text(encoding="utf-8"),
            )
            self.assertIn("learning_rate_path", paths)

    def test_write_rfdetr_training_report_omits_lr_plot_without_lr(self) -> None:
        """Only write learning-rate charts when RF-DETR LR columns exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics_path = root / "metrics.csv"
            metrics_path.write_text(
                "\n".join(
                    [
                        "epoch,step,val/precision,val/recall,val/F1",
                        "0,1,0.7,0.8,0.7467",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            paths = write_rfdetr_training_report(
                metrics_csv_path=metrics_path,
                output_dir=root / "report",
                run_id="run-a",
            )

            self.assertNotIn("learning_rate_path", paths)

    def test_parse_rfdetr_training_run_spec_accepts_plain_and_labeled(self) -> None:
        """Parse RF-DETR training comparison labels from label=path values."""
        plain = parse_rfdetr_training_run_spec("runs/training-reports/run-a")
        labeled = parse_rfdetr_training_run_spec("EMA1=runs/training-reports/run-a")

        self.assertEqual(plain.path, Path("runs/training-reports/run-a"))
        self.assertIsNone(plain.display_label)
        self.assertEqual(labeled.path, Path("runs/training-reports/run-a"))
        self.assertEqual(labeled.display_label, "EMA1")

    def test_write_rfdetr_training_comparison_reports_writes_outputs(self) -> None:
        """Compare RF-DETR training reports and rank by validation F2."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_metrics_path = root / "first_metrics.csv"
            second_metrics_path = root / "second_metrics.csv"
            write_rfdetr_metrics_fixture(first_metrics_path, second_recall=0.8)
            write_rfdetr_metrics_fixture(second_metrics_path, second_recall=0.9)
            first_report = root / "first"
            second_report = root / "second"
            write_rfdetr_training_report(first_metrics_path, first_report, "first")
            write_rfdetr_training_report(second_metrics_path, second_report, "second")

            paths = write_rfdetr_training_comparison_reports(
                training_run_specs=[
                    parse_rfdetr_training_run_spec(f"EMA1={first_report}"),
                    second_report,
                ],
                output_dir=root / "comparison",
            )

            summary_csv = paths["summary_csv_path"].read_text(encoding="utf-8")
            summary_markdown = paths["summary_markdown_path"].read_text(
                encoding="utf-8",
            )
            self.assertIn("second", summary_csv.splitlines()[1])
            self.assertIn("EMA1", summary_csv)
            self.assertIn("not benchmark accuracy reporting", summary_markdown)
            self.assertIn(
                "RF-DETR Validation F2 Comparison",
                paths["validation_f2_path"].read_text(encoding="utf-8"),
            )
            self.assertIn("learning_rate_path", paths)

    def test_load_rfdetr_training_runs_requires_two_runs(self) -> None:
        """Require at least two RF-DETR training runs for comparison."""
        with self.assertRaisesRegex(ValueError, "At least two"):
            load_rfdetr_training_runs([Path("runs/training-reports/run-a")])


def threshold_validation_result_for_test(
    model_name: str,
    selected_threshold: float,
    precision: float,
    recall: float,
    f1: float,
    f2: float,
    threshold_metrics: list[dict[str, float | int]] | None = None,
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
        threshold_metrics=threshold_metrics
        or [
            threshold_row_for_test(0.1, 0.7, 0.7),
            selected_metrics,
        ],
    )


def threshold_row_for_test(
    confidence_threshold: float,
    precision: float,
    recall: float,
) -> dict[str, float | int]:
    """Build one threshold metrics row for report tests."""
    return {
        "confidence_threshold": confidence_threshold,
        "true_positive_count": int(round(recall * 10)),
        "false_positive_count": 3,
        "false_negative_count": int(round((1 - recall) * 10)),
        "precision": precision,
        "recall": recall,
        "f1": 0.0,
        "f2": 0.0,
    }


def write_rfdetr_metrics_fixture(path: Path, second_recall: float = 0.8) -> None:
    """Write a sparse RF-DETR metrics.csv fixture."""
    rows = [
        [
            "epoch",
            "step",
            "train/loss",
            "train/lr",
            "val/loss",
            "val/precision",
            "val/recall",
            "val/F1",
            "val/mAP_50",
            "val/mAP_50_95",
        ],
        ["0", "10", "", "0.001", "", "", "", "", "", ""],
        ["0", "20", "", "", "4.0", "0.9", "0.75", "0.8182", "0.7", "0.4"],
        ["0", "20", "6.0", "0.001", "", "", "", "", "", ""],
        [
            "1",
            "30",
            "",
            "",
            "3.0",
            "0.8",
            str(second_recall),
            "0.8182",
            "0.8",
            "0.5",
        ],
        ["1", "30", "5.0", "0.0005", "", "", "", "", "", ""],
    ]
    path.write_text(
        "\n".join(",".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write newline-delimited JSON fixture rows."""
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def json_dumps(payload: dict) -> str:
    """Serialize compact JSON for test fixture files."""
    return json.dumps(payload, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
