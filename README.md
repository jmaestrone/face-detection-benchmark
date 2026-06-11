# Face Detection Benchmark

Local tooling for building, reviewing, training, validating, and evaluating target-domain face detection workflows from trimmed Instawork videos.

The repo started as a simple benchmark runner, but now covers frame extraction, RF-DETR pre-labeling, Roboflow review batches, cleaned Roboflow dataset downloads, threshold validation, final benchmark evaluation, and RF-DETR training/report analysis.

The CLI tools in `src/face_detection_benchmark/` are the source of truth. Notebooks are guided wrappers around those commands.

## Setup

This repo uses `uv` and Python 3.12.

```bash
uv sync
```

InsightFace/SCRFD prediction support is optional because it adds ONNX Runtime and InsightFace dependencies:

```bash
uv sync --extra insightface
```

EgoBlur prediction support is optional because it adds the EgoBlur package and its TorchScript runtime dependencies:

```bash
uv sync --extra egoblur
```

RF-DETR inference requires a local checkpoint. Put model files under `models/`; for example:

```text
models/checkpoint_best_ema.pth
```

EgoBlur Gen2 face benchmark inference requires the downloaded face model under:

```text
models/egoblur/ego_blur_face_gen2.jit
```

Large local artifacts are intentionally ignored by git.

## Artifact Layout

```text
face-trimmed-videos/      # source trimmed videos, ignored
models/                   # local RF-DETR checkpoints, ignored
models/egoblur/           # local EgoBlur TorchScript models, ignored
data/frames/              # extracted image frames, ignored
data/predictions/         # raw RF-DETR pre-label detections, ignored
data/roboflow-export/     # Roboflow-ready COCO review exports, ignored
data/benchmark/           # downloaded benchmark or validation datasets, ignored
data/training/            # training datasets, ignored
data/training/rfdetr/     # default RF-DETR training datasets, ignored
notebooks/                # local guided workflow wrappers
runs/                     # previews, logs, reports, and experiments, ignored
runs/training/            # RF-DETR training outputs and checkpoints, ignored
```

Generated artifacts should stay under ignored local paths. Do not commit source videos, model files, downloaded Roboflow datasets, predictions, reports, or training outputs.

## Workflow Map

- **Dataset policy**: read [`docs/dataset-policy.md`](docs/dataset-policy.md) before deciding whether a labeled dataset is training, validation, final test, or a Roboflow pre-label batch.
- **Roboflow workflows**: use [`docs/roboflow-workflows.md`](docs/roboflow-workflows.md) for frame extraction, RF-DETR pre-labeling, COCO review exports, per-video manual review batches, cleaned benchmark downloads, and `1 fps` vs `3 fps` conventions.
- **Evaluation workflows**: use [`docs/evaluation-workflows.md`](docs/evaluation-workflows.md) for RF-DETR, InsightFace, and EgoBlur prediction, threshold validation, final benchmark evaluation, latency reports, and prediction overlays.
- **RF-DETR training**: use [`docs/rfdetr-training.md`](docs/rfdetr-training.md) for training datasets, `train-rfdetr`, training telemetry reports, and trained checkpoint validation/evaluation.

## Dataset Policy Summary

Keep the data lanes separate:

- `data/training/` is for model training datasets.
- `runs/validation/` is for threshold, checkpoint, and model-selection analysis.
- `data/benchmark/target-video-test-3fps-clean/test/` is the canonical cleaned final test benchmark when it has not been used for selection.
- `data/frames/`, `data/predictions/`, and `data/roboflow-export/` are temporary pre-label/review-batch artifacts.

If the cleaned target-domain dataset is used to choose thresholds, checkpoints, or model variants, treat that output as validation analysis, not as unbiased final benchmark evidence.

Code-backed safeguards:

- `train-rfdetr` requires an explicit training dataset and rejects paths under `data/benchmark/`.
- `download-roboflow-benchmark` validates the cleaned test-only Roboflow export, category `Human face`, and the expected image count for the current dataset.

## Notebooks

- `notebooks/rfdetr_workflow.ipynb`: RF-DETR pre-labeling and Roboflow review batch workflow.
- `notebooks/rfdetr_training_workflow.ipynb`: RF-DETR training wrapper.
- `notebooks/insightface_workflow.ipynb`: InsightFace/SCRFD prediction and threshold-validation wrapper.

Keep notebooks as guided local wrappers. Prefer documenting durable policy and workflow details in `docs/`.
