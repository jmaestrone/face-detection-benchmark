# Roboflow Workflows

Use this guide for frame extraction, RF-DETR pre-labeling, Roboflow review batch exports, manual review, and cleaned benchmark downloads.

## Frame Extraction

Source trimmed videos live under:

```text
face-trimmed-videos/
```

Extract sampled frames:

```bash
uv run face-benchmark extract-frames
```

By default this reads `face-trimmed-videos/`, writes to `data/frames/`, and samples at `1 fps` from `DEFAULT_FRAME_FPS`. Use `--fps` for denser or sparser sampling, or `--every-n-frames` for frame-index-based sampling.

For the current 12 trimmed videos, the default `1 fps` extraction produces 61 JPEG frames plus:

```text
data/frames/metadata.jsonl
```

## Sampling Conventions

Use sampling rate as part of the dataset/review-batch identity.

- Use `1 fps` for lighter manual review batches, quick iteration, or smoke checks.
- Use `3 fps` when the goal is denser target-domain benchmark coverage.
- Encode the sampling rate in batch and dataset names, such as `target-video-test-3fps-clean`.

The current reviewed benchmark convention is:

```text
target-video-test-3fps-clean
```

The current cleaned benchmark download expects 169 images in the Roboflow `test` split.

## RF-DETR Pre-labeling

Run local RF-DETR on extracted frames:

```bash
uv run face-benchmark predict-faces --weights models/checkpoint_best_ema.pth
```

For a quick smoke run:

```bash
uv run face-benchmark predict-faces \
  --weights models/checkpoint_best_ema.pth \
  --limit 1 \
  --preview-dir runs/previews
```

By default, detections are written to:

```text
data/predictions/predictions.jsonl
```

These detections are pre-labels for review. They are not final ground truth until corrected and accepted in Roboflow.

## Roboflow Review Batch Export

Export frame images and RF-DETR detections as a Roboflow-ready COCO dataset:

```bash
uv run face-benchmark export-coco
```

By default this writes under:

```text
data/roboflow-export/
```

The export includes only frames with at least one RF-DETR detection unless `--include-empty` is provided. Use `--include-empty` when no-face frames should be preserved for review and evaluation.

## Per-video Review Batches

Keep per-video Roboflow upload/review batches distinct. Use names that include:

- source video or video group,
- review purpose,
- sampling rate,
- date or sequence number when useful.

Example names:

```text
video-03-review-1fps-2026-06
target-video-test-3fps-review-batch-01
```

Keep the local exports under ignored paths such as `data/roboflow-export/` or another ignored `data/` subdirectory. Do not commit exported images, annotations, Roboflow downloads, or review artifacts.

The `upload-roboflow` command is currently a placeholder:

```bash
uv run face-benchmark upload-roboflow
```

Until upload support is implemented, upload and manual review steps are partly manual in Roboflow.

## Roboflow Manual Review

In Roboflow, review the uploaded pre-label batch before it becomes a cleaned dataset:

- correct noisy RF-DETR boxes,
- add missing face boxes,
- remove false-positive boxes,
- preserve intended no-face frames when they are part of the batch policy,
- keep final benchmark images in the Roboflow `test` split,
- use no augmentations for the final benchmark version.

Do not mix review batches intended for training, validation, and final test unless the resulting dataset version has an explicit policy and split design.

## Cleaned Dataset Download

Copy the local environment template and fill in private Roboflow values:

```bash
cp .env.example .env
```

Required values:

```text
ROBOFLOW_API_KEY=...
ROBOFLOW_WORKSPACE=...
ROBOFLOW_PROJECT=...
ROBOFLOW_VERSION=1
```

Do not commit or paste these values into notebooks, docs, or source files.

Download and validate the cleaned benchmark:

```bash
uv run face-benchmark download-roboflow-benchmark --overwrite
```

You can also pass the private source explicitly:

```bash
uv run face-benchmark download-roboflow-benchmark \
  --workspace <workspace> \
  --project <project> \
  --version <version> \
  --overwrite
```

By default this writes to:

```text
data/benchmark/target-video-test-3fps-clean/
```

The command validates that the expected split contains the expected image count, uses category `Human face`, and has no non-empty `train` or `valid` split. It also writes non-secret source metadata to:

```text
data/benchmark/target-video-test-3fps-clean/roboflow_source.json
```
