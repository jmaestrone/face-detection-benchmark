"""Shared configuration defaults for local pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

DEFAULT_VIDEO_DIR = Path("face-trimmed-videos")
DEFAULT_MODEL_PATH = Path("models/checkpoint_best_ema.pth")
DEFAULT_FRAMES_DIR = Path("data/frames")
DEFAULT_PREDICTIONS_DIR = Path("data/predictions")
DEFAULT_PREDICTIONS_PATH = DEFAULT_PREDICTIONS_DIR / "predictions.jsonl"
DEFAULT_ROBOFLOW_EXPORT_DIR = Path("data/roboflow-export")
DEFAULT_BENCHMARK_DATA_DIR = Path("data/benchmark")
DEFAULT_BENCHMARK_DATASET_NAME = "target-video-test-3fps-clean"
DEFAULT_RUNS_DIR = Path("runs")

DEFAULT_FRAME_FPS = 1.0
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
FACE_CATEGORY_NAME = "Human face"

DEFAULT_ROBOFLOW_FORMAT = "coco"
DEFAULT_ROBOFLOW_TEST_SPLIT = "test"
DEFAULT_BENCHMARK_IMAGE_COUNT = 169
