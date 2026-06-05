"""Tests for inference helper functions."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from face_detection_benchmark.inference import (
    iter_batches,
    load_frame_image_batch,
    load_image_batch,
    predict_faces_from_coco_dataset,
    predict_faces_from_frames,
    rfdetr_model_name_from_weights,
)
from face_detection_benchmark.models.insightface import (
    DEFAULT_INSIGHTFACE_CTX_ID,
    DEFAULT_INSIGHTFACE_DET_SIZE,
    DEFAULT_INSIGHTFACE_MODEL_PACK,
    DEFAULT_INSIGHTFACE_PROVIDERS,
    DEFAULT_INSIGHTFACE_THRESHOLD,
    InsightFaceConfig,
    InsightFaceDetector,
    detector_output_to_records,
    parse_providers,
)


class InferenceHelpersTest(unittest.TestCase):
    """Coverage for lightweight inference helpers."""

    def test_rfdetr_model_name_from_weights(self) -> None:
        """Build a readable model name from an RF-DETR checkpoint path."""
        self.assertEqual(
            rfdetr_model_name_from_weights(Path("models/checkpoint_best_ema_2.pth")),
            "rfdetr-checkpoint-best-ema-2",
        )

    def test_public_image_loader_alias_remains_compatible(self) -> None:
        """Keep the legacy public image batch loader alias available."""
        self.assertIs(load_image_batch, load_frame_image_batch)

    def test_iter_batches_yields_fixed_size_groups(self) -> None:
        """Split records into fixed-size batches without dropping leftovers."""
        self.assertEqual(
            list(iter_batches([1, 2, 3, 4, 5], batch_size=2)),
            [[1, 2], [3, 4], [5]],
        )

    def test_frame_prediction_validates_inputs_before_loading_model(self) -> None:
        """Reject missing frame inputs before constructing the RF-DETR adapter."""
        with self.assertRaisesRegex(ValueError, "Frames directory does not exist"):
            predict_faces_from_frames(
                frames_dir=Path("missing-frames"),
                weights_path=Path("missing-weights.pth"),
            )

    def test_benchmark_prediction_validates_inputs_before_loading_model(self) -> None:
        """Reject missing benchmark inputs before constructing the RF-DETR adapter."""
        with self.assertRaisesRegex(ValueError, "Dataset directory does not exist"):
            predict_faces_from_coco_dataset(
                dataset_dir=Path("missing-dataset"),
                weights_path=Path("missing-weights.pth"),
            )

    def test_parse_insightface_providers(self) -> None:
        """Parse comma-separated ONNX Runtime provider names."""
        self.assertEqual(
            parse_providers("CUDAExecutionProvider, CPUExecutionProvider"),
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
        self.assertEqual(
            parse_providers(["CUDAExecutionProvider", " "]),
            ("CUDAExecutionProvider",),
        )

    def test_insightface_detection_output_to_records(self) -> None:
        """Normalize InsightFace detector output boxes into prediction records."""
        records = detector_output_to_records(
            (
                [
                    [1, 2, 11, 22, 0.9876543],
                    [3, 4, 5, 6, 0.1],
                ],
                None,
            )
        )

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].bbox_xyxy, [1.0, 2.0, 11.0, 22.0])
        self.assertEqual(records[0].bbox_xywh, [1.0, 2.0, 10.0, 20.0])
        self.assertEqual(records[0].confidence, 0.987654)
        self.assertIsNone(records[0].class_id)
        self.assertEqual(detector_output_to_records(None), [])

    def test_insightface_missing_dependency_message(self) -> None:
        """Explain how to install the optional InsightFace dependency."""
        real_import = __import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name.startswith("insightface"):
                raise ImportError("missing insightface")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ValueError, "uv sync --extra insightface"):
                InsightFaceDetector(
                    InsightFaceConfig(
                        model_pack=DEFAULT_INSIGHTFACE_MODEL_PACK,
                        providers=DEFAULT_INSIGHTFACE_PROVIDERS,
                        ctx_id=DEFAULT_INSIGHTFACE_CTX_ID,
                        det_size=DEFAULT_INSIGHTFACE_DET_SIZE,
                        threshold=DEFAULT_INSIGHTFACE_THRESHOLD,
                    )
                )


if __name__ == "__main__":
    unittest.main()
