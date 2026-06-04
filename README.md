# RF-DETR Faces Tests

Local tooling for building a face-detection dataset from trimmed videos. The
pipeline extracts sampled frames, runs a local RF-DETR face checkpoint, exports
COCO annotations, and can upload the resulting dataset to Roboflow.

The current notebook, `RF_DETR_face_det.ipynb`, is a reference from the Colab
workflow. The local scripts should become the source of truth, with the notebook
kept as an exploratory wrapper around those scripts.

## Setup

This repo uses `uv` and Python 3.12.

```bash
uv sync
```

RF-DETR inference requires a local checkpoint. Put model files under `models/`;
for example:

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
runs/                     # previews, logs, and experiments, ignored
```

## Planned Pipeline

Extract sampled frames from the current videos:

```bash
uv run rfdetr-faces extract-frames
```

Run RF-DETR face detection with a local checkpoint:

```bash
uv run rfdetr-faces predict-faces --weights models/checkpoint_best_ema.pth
```

Export detections as COCO ground-truth annotations for Roboflow:

```bash
uv run rfdetr-faces export-coco
```

Upload the COCO dataset to Roboflow:

```bash
export ROBOFLOW_API_KEY=...
uv run rfdetr-faces upload-roboflow --workspace <workspace> --project <project-id>
```

The exported RF-DETR detections are treated as ground truth for upload. Any noisy
or missing boxes should be corrected later in Roboflow.
