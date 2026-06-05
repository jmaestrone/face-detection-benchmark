"""Tests for prediction overlay rendering reports and CLI wiring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from face_detection_benchmark.cli import app
from face_detection_benchmark.reports import (
    PredictionOverlaySpec,
    parse_prediction_overlay_spec,
    render_prediction_overlays,
)
from typer.testing import CliRunner


class PredictionOverlayTest(unittest.TestCase):
    """Coverage for prediction overlay spec parsing and rendering."""

    def test_parse_prediction_overlay_spec(self) -> None:
        """Parse label, JSONL path, and threshold from one CLI spec."""
        spec = parse_prediction_overlay_spec("model-a=predictions/model-a.jsonl:0.35")

        self.assertEqual(spec.label, "model-a")
        self.assertEqual(spec.predictions_path, Path("predictions/model-a.jsonl"))
        self.assertEqual(spec.confidence_threshold, 0.35)

    def test_parse_prediction_overlay_spec_rejects_invalid_threshold(self) -> None:
        """Reject malformed thresholds before rendering starts."""
        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            parse_prediction_overlay_spec("model-a=predictions.jsonl:1.5")

    def test_parse_prediction_overlay_spec_rejects_unsafe_label(self) -> None:
        """Reject labels that could escape the per-model output directory."""
        with self.assertRaisesRegex(ValueError, "must start with a letter or number"):
            parse_prediction_overlay_spec("../model=predictions.jsonl:0.5")

    def test_render_prediction_overlays_writes_model_comparison_and_summaries(
        self,
    ) -> None:
        """Write per-model overlays, comparison images, and summary artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = write_overlay_dataset(root)
            first_predictions_path = write_predictions(
                root / "first.jsonl",
                model_name="first-model",
                detections=[
                    detection([4, 4, 12, 12], 0.9),
                    detection([24, 4, 36, 16], 0.8),
                    detection([4, 24, 12, 36], 0.4),
                ],
            )
            second_predictions_path = write_predictions(
                root / "second.jsonl",
                model_name="second-model",
                detections=[],
            )

            result = render_prediction_overlays(
                dataset_dir=dataset_dir,
                prediction_specs=[
                    PredictionOverlaySpec("first", first_predictions_path, 0.5),
                    PredictionOverlaySpec("second", second_predictions_path, 0.5),
                ],
                output_dir=root / "visualizations",
                iou_threshold=0.5,
            )

            first_overlay_path = result.output_dir / "models" / "first" / "image.png"
            comparison_path = result.output_dir / "comparison" / "image.png"
            self.assertTrue(first_overlay_path.exists())
            self.assertTrue(comparison_path.exists())
            first_overlay = read_image(first_overlay_path)
            comparison_overlay = read_image(comparison_path)
            self.assertEqual(first_overlay.shape[:2], (48, 48))
            self.assertEqual(comparison_overlay.shape[:2], (76, 96))
            self.assertEqual(first_overlay[4, 4].tolist(), [0, 255, 0])
            self.assertEqual(first_overlay[4, 24].tolist(), [0, 255, 255])
            self.assertEqual(first_overlay[24, 4].tolist(), [0, 0, 255])

            summary_csv = result.summary_csv_path.read_text(encoding="utf-8")
            self.assertIn("label,model_name,prediction_path,threshold", summary_csv)
            self.assertIn("first,first-model", summary_csv)
            summary_json = json.loads(result.summary_json_path.read_text("utf-8"))
            self.assertEqual(summary_json["iou_threshold"], 0.5)
            self.assertEqual(summary_json["models"][0]["true_positive_count"], 1)
            self.assertEqual(summary_json["models"][0]["false_positive_count"], 1)
            self.assertEqual(summary_json["models"][0]["false_negative_count"], 1)
            self.assertEqual(summary_json["models"][0]["precision"], 0.5)
            self.assertEqual(summary_json["models"][0]["recall"], 0.5)

    def test_render_prediction_overlays_cli_uses_run_id_output_dir(self) -> None:
        """Invoke the registered CLI command with the requested command name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = write_overlay_dataset(root)
            predictions_path = write_predictions(
                root / "predictions.jsonl",
                model_name="cli-model",
                detections=[detection([4, 4, 12, 12], 0.9)],
            )

            result = CliRunner().invoke(
                app,
                [
                    "render-prediction-overlays",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--runs-dir",
                    str(root / "runs"),
                    "--run-id",
                    "cli-overlays",
                    "--prediction-spec",
                    f"cli={predictions_path}:0.5",
                    "--iou-threshold",
                    "0.5",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            output_dir = root / "runs" / "visualizations" / "cli-overlays"
            self.assertTrue((output_dir / "models" / "cli" / "image.png").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
            self.assertIn("Rendered overlays for 1 model", result.output)


def write_overlay_dataset(root: Path) -> Path:
    """Write one small COCO dataset with a real PNG image."""
    dataset_dir = root / "dataset"
    dataset_dir.mkdir()
    write_blank_image(dataset_dir / "image.png")
    (dataset_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "image.png",
                        "width": 48,
                        "height": 48,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [4, 4, 8, 8],
                        "area": 64,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [4, 24, 8, 12],
                        "area": 96,
                        "iscrowd": 0,
                    },
                ],
                "categories": [{"id": 1, "name": "Human face"}],
            }
        ),
        encoding="utf-8",
    )
    return dataset_dir


def write_blank_image(image_path: Path) -> None:
    """Write a deterministic white PNG fixture."""
    import cv2
    import numpy as np

    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)


def write_predictions(
    predictions_path: Path,
    model_name: str,
    detections: list[dict],
) -> Path:
    """Write one normalized prediction JSONL file."""
    predictions_path.write_text(
        json.dumps(
            {
                "file_name": "image.png",
                "width": 48,
                "height": 48,
                "model_name": model_name,
                "detections": detections,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return predictions_path


def detection(bbox_xyxy: list[float], confidence: float) -> dict:
    """Build one normalized JSON detection payload."""
    x_min, y_min, x_max, y_max = bbox_xyxy
    return {
        "bbox_xyxy": bbox_xyxy,
        "bbox_xywh": [x_min, y_min, x_max - x_min, y_max - y_min],
        "confidence": confidence,
        "class_id": 1,
        "class_name": "Human face",
    }


def read_image(image_path: Path):
    """Read a rendered image with OpenCV."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise AssertionError(f"Could not read image: {image_path}")
    return image


if __name__ == "__main__":
    unittest.main()
