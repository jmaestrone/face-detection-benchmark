"""Tests for inference helper functions."""

from __future__ import annotations

import unittest
from pathlib import Path

from face_detection_benchmark.inference import rfdetr_model_name_from_weights


class InferenceHelpersTest(unittest.TestCase):
    """Coverage for lightweight inference helpers."""

    def test_rfdetr_model_name_from_weights(self) -> None:
        """Build a readable model name from an RF-DETR checkpoint path."""
        self.assertEqual(
            rfdetr_model_name_from_weights(Path("models/checkpoint_best_ema_2.pth")),
            "rfdetr-checkpoint-best-ema-2",
        )


if __name__ == "__main__":
    unittest.main()
