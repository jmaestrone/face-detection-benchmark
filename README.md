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
data/benchmark/           # downloaded benchmark datasets, ignored
notebooks/                # local notebooks and Colab references
runs/                     # previews, logs, and experiments, ignored
```

`notebooks/rfdetr_workflow.ipynb` is the local RF-DETR workflow notebook. The CLI tools are the source of truth for the local pipeline; the notebook is a guided wrapper around those tools.

## Dataset Policy

The cleaned Roboflow dataset created from these target videos is a test-only benchmark. Keep all uploaded images in the Roboflow `test` split, use no augmentations for the benchmark version, and do not train, fine-tune, or tune thresholds on this dataset.

## Benchmark Dataset Download

The canonical cleaned target-domain benchmark should be downloaded from the private Roboflow project version. Keep workspace, project, and API-key values out of committed files.

Copy the example environment file and fill in private values locally:

```bash
cp .env.example .env
```

Required `.env` values:

```text
ROBOFLOW_API_KEY=...
ROBOFLOW_WORKSPACE=...
ROBOFLOW_PROJECT=...
ROBOFLOW_VERSION=1
```

Do not commit or paste these values into notebooks, docs, or source files. The `.env` file is ignored by git.

Download and validate the benchmark:

```bash
uv run face-benchmark download-roboflow-benchmark --overwrite
```

You can also pass the private Roboflow source explicitly instead of using `.env`:

```bash
uv run face-benchmark download-roboflow-benchmark \
  --workspace <workspace> \
  --project <project> \
  --version <version> \
  --overwrite
```

For the current cleaned benchmark, keep the local dataset name as:

```text
target-video-test-3fps-clean
```

By default this writes to `data/benchmark/target-video-test-3fps-clean/` and validates that the `test` split contains 169 images, uses category `Human face`, and has no non-empty `train` or `valid` split. It also writes non-secret source metadata to `roboflow_source.json`.

## Benchmark Evaluation

Evaluate normalized prediction JSONL files against the cleaned COCO test split:

```bash
uv run face-benchmark evaluate-detections \
  --predictions-path runs/benchmarks/<run-id>/predictions/<model>.jsonl \
  --confidence-threshold <preselected-threshold>
```

By default this reads `data/benchmark/target-video-test-3fps-clean/test/`, writes metrics under `runs/benchmarks/<new-run-id>/`, appends one row to the ignored local comparison table at `runs/benchmarks/results.csv`, and regenerates a human-readable Markdown leaderboard at `runs/benchmarks/results.md`. Prediction rows must use `file_name` values that match the COCO test split image filenames. The evaluator reports precision, recall, F1, F2 at IoU 0.50, `AP50`, `AP75`, and `mAP@[0.50:0.95]`.

Choose `--confidence-threshold` before evaluating the benchmark. Do not run multiple thresholds on the test set to choose the best one. A diagnostic confidence sweep is available with `--include-confidence-sweep`, but it should not be used for threshold selection on this test-only dataset.

## Threshold Validation

Generate RF-DETR predictions directly on the cleaned COCO split before validation:

```bash
uv run face-benchmark predict-rfdetr-benchmark \
  --weights models/checkpoint_best_ema.pth \
  --run-id rfdetr-ema-1
```

This command defaults to a low inference threshold of `0.005` so validation sweeps can compare operating thresholds without being limited by an earlier `0.25` prediction filter.

If you intentionally treat a labeled split as validation data, choose an operating threshold with:

```bash
uv run face-benchmark validate-thresholds \
  --predictions-path runs/benchmarks/<run-id>/predictions/<model>.jsonl \
  --selection-metric f2
```

By default this reads the same local COCO split path as benchmark evaluation, evaluates thresholds `0.005`, `0.01`, and `0.05` through `0.80` in `0.05` steps at IoU `0.50`, and writes validation-only artifacts under `runs/validation/<new-run-id>/`:

- `threshold_validation.json`
- `selected_threshold.json`
- `threshold_metrics.csv`
- `threshold_metrics.md`
- `precision_recall.svg`
- `f1_f2_by_threshold.svg`

Use `--selection-metric f1`, `--selection-metric f2`, `--selection-metric precision`, or `--selection-metric recall` to decide how the selected threshold is chosen. If multiple thresholds have the same selected metric value, the higher threshold is selected. Use `--thresholds 0.10,0.20,0.30` to provide a custom threshold grid. If this cleaned target-domain dataset is used to choose a threshold, treat the output as validation analysis rather than an unbiased test benchmark result.

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

The Roboflow upload command is currently a planned placeholder:

```bash
uv run face-benchmark upload-roboflow
```

When upload support is implemented, the exported RF-DETR detections should be treated as ground truth for upload. Any noisy or missing boxes should be corrected later in Roboflow.
