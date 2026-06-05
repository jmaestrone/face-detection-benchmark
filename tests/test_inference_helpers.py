"""Tests for inference helper functions."""

from __future__ import annotations

import unittest
from pathlib import Path

from face_detection_benchmark.inference import (
    iter_batches,
    load_frame_image_batch,
    load_image_batch,
    predict_faces_from_coco_dataset,
    predict_faces_from_frames,
    rfdetr_model_name_from_weights,
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


if __name__ == "__main__":
    unittest.main()
