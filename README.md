# Face Detection Benchmark

Local tooling for building and evaluating a target-domain face detection benchmark from trimmed videos. The current pipeline extracts sampled frames, runs a local RF-DETR face checkpoint, exports COCO annotations, and supports Roboflow-based benchmark dataset creation.

## Setup

This repo uses `uv` and Python 3.12.

```bash
uv sync
```

RF-DETR inference requires a local checkpoint. Put model files under `models/`; for example:

```text
models/checkpoint_best_ema.pth
```

Large local artifacts are intentionally ignored by git.

## Artifact Layout

```text
face-trimmed-videos/      # current source videos
models/                   # local RF-DETR checkpoints, ignored
data/frames/              # extracted image frames, ignored
data/predictions/         # raw RF-DETR detections, ignored
data/roboflow-export/     # COCO dataset export, ignored
notebooks/                # local notebooks and Colab references
runs/                     # previews, logs, and experiments, ignored
```

`notebooks/rfdetr_workflow.ipynb` is the local RF-DETR workflow notebook. The CLI tools are the source of truth for the local pipeline; the notebook is a guided wrapper around those tools.

## Dataset Policy

The cleaned Roboflow dataset created from these target videos is a test-only benchmark. Keep all uploaded images in the Roboflow `test` split, use no augmentations for the benchmark version, and do not train, fine-tune, or tune thresholds on this dataset.

## RF-DETR Labeling Pipeline

Extract sampled frames from the current videos:

```bash
uv run face-benchmark extract-frames
```

By default this reads `face-trimmed-videos/`, writes images to `data/frames/`, and samples at `1 fps`. Use `--fps` for a denser or sparser sample, or `--every-n-frames` when you want frame-index based sampling.

For the current 12 trimmed videos, the default `1 fps` extraction produces 61 JPEG frames plus `data/frames/metadata.jsonl`.

Run RF-DETR face detection with a local checkpoint:

```bash
uv run face-benchmark predict-faces --weights models/checkpoint_best_ema.pth
```

For a quick smoke run on one extracted frame:

```bash
uv run face-benchmark predict-faces \
  --weights models/checkpoint_best_ema.pth \
  --limit 1 \
  --preview-dir runs/previews
```

Export detections as COCO ground-truth annotations for Roboflow:

```bash
uv run face-benchmark export-coco
```

By default this exports only frames that have at least one RF-DETR detection. Use `--include-empty` if you also want no-detection frames in the COCO dataset.

Upload the COCO dataset to Roboflow:

```bash
export ROBOFLOW_API_KEY=...
uv run face-benchmark upload-roboflow --workspace <workspace> --project <project-id>
```

The exported RF-DETR detections are treated as ground truth for upload. Any noisy or missing boxes should be corrected later in Roboflow.
