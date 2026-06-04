# RF-DETR Faces Tests

Local tooling for building a face-detection dataset from trimmed videos. The pipeline extracts sampled frames, runs a local RF-DETR face checkpoint, exports COCO annotations, and can upload the resulting dataset to Roboflow.

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

`notebooks/RF_DETR_face_det.ipynb` is the original Colab reference notebook. The CLI tools are the source of truth for the local pipeline; the notebook will be updated later to run those tools locally on a Mac.

## Planned Pipeline

Extract sampled frames from the current videos:

```bash
uv run rfdetr-faces extract-frames
```

By default this reads `face-trimmed-videos/`, writes images to `data/frames/`, and samples at `1 fps`. Use `--fps` for a denser or sparser sample, or `--every-n-frames` when you want frame-index based sampling.

For the current 12 trimmed videos, the default `1 fps` extraction produces 61 JPEG frames plus `data/frames/metadata.jsonl`.

Run RF-DETR face detection with a local checkpoint:

```bash
uv run rfdetr-faces predict-faces --weights models/checkpoint_best_ema.pth
```

For a quick smoke run on one extracted frame:

```bash
uv run rfdetr-faces predict-faces \
  --weights models/checkpoint_best_ema.pth \
  --limit 1 \
  --preview-dir runs/previews
```

Export detections as COCO ground-truth annotations for Roboflow:

```bash
uv run rfdetr-faces export-coco
```

By default this exports only frames that have at least one RF-DETR detection. Use `--include-empty` if you also want no-detection frames in the COCO dataset.

Upload the COCO dataset to Roboflow:

```bash
export ROBOFLOW_API_KEY=...
uv run rfdetr-faces upload-roboflow --workspace <workspace> --project <project-id>
```

The exported RF-DETR detections are treated as ground truth for upload. Any noisy or missing boxes should be corrected later in Roboflow.
