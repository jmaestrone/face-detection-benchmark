# Face Detection Benchmark

Local tooling for building and evaluating a target-domain face detection benchmark from trimmed videos. The current pipeline extracts sampled frames, runs a local RF-DETR face checkpoint, exports COCO annotations, and supports Roboflow-based benchmark dataset creation.

## Setup

This repo uses `uv` and Python 3.12.

```bash
uv sync
```

InsightFace/SCRFD prediction support is optional because it adds ONNX Runtime
and InsightFace dependencies:

```bash
uv sync --extra insightface
```

EgoBlur prediction support is optional because it adds the EgoBlur package and
its TorchScript runtime dependencies:

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
face-trimmed-videos/      # current source videos
models/                   # local RF-DETR checkpoints, ignored
models/egoblur/           # local EgoBlur TorchScript models, ignored
data/frames/              # extracted image frames, ignored
data/predictions/         # raw RF-DETR detections, ignored
data/roboflow-export/     # COCO dataset export, ignored
data/benchmark/           # downloaded benchmark datasets, ignored
data/training/            # training datasets, ignored
data/training/rfdetr/     # RF-DETR training datasets, ignored
notebooks/                # local notebooks and Colab references
runs/                     # previews, logs, and experiments, ignored
runs/training/            # training outputs and checkpoints, ignored
```

`notebooks/rfdetr_workflow.ipynb` is the local RF-DETR labeling workflow notebook, `notebooks/rfdetr_training_workflow.ipynb` is the RF-DETR training wrapper, and `notebooks/insightface_workflow.ipynb` is the local InsightFace/SCRFD benchmark wrapper. The CLI tools are the source of truth for the local pipeline; notebooks are guided wrappers around those tools.

## Dataset Policy

The cleaned Roboflow dataset created from these target videos is a test-only benchmark. Keep all uploaded images in the Roboflow `test` split, use no augmentations for the benchmark version, and do not train, fine-tune, or tune thresholds on this dataset.

Benchmark data and training data must stay separate:

- `data/benchmark/` is for downloaded benchmark or validation datasets only.
- `data/training/` is for model training datasets only.
- `data/training/rfdetr/` is the default local root for RF-DETR training datasets.
- `runs/training/` is the default local root for RF-DETR training outputs, metadata, and checkpoints.

Do not use `data/benchmark/target-video-test-3fps-clean/` for RF-DETR training, fine-tuning, augmentation experiments, or threshold tuning if the dataset is being treated as test data. RF-DETR training support must require an explicit training dataset directory and must refuse paths under `data/benchmark/`.

## RF-DETR Training

RF-DETR training uses the CLI as the source of truth. Keep training datasets under `data/training/`, keep training outputs under `runs/training/`, and never point the training command at `data/benchmark/`.

The installed RF-DETR package exposes training through `RFDETRLarge.train(...)`. Training dependencies are optional in RF-DETR and may not be present in a default local environment. If a training run reports missing RF-DETR training dependencies, install the RF-DETR train extras in the local environment before running a real training job.

Run training from an explicit training dataset directory:

```bash
uv run face-benchmark train-rfdetr \
  --dataset-dir data/training/rfdetr \
  --output-dir runs/training/<run-id> \
  --epochs 100 \
  --batch-size 4 \
  --device auto
```

By default the command uses RF-DETR's `roboflow` dataset format. Use `--dataset-file coco`, `--dataset-file yolo`, or `--dataset-file o365` only when the training dataset matches that RF-DETR format. Use `--weights models/<checkpoint>.pth` when fine-tuning from a local checkpoint.

The command writes reproducibility artifacts before training starts:

```text
runs/training/<run-id>/config.json
runs/training/<run-id>/metadata.json
```

RF-DETR writes its own training outputs and checkpoints under the same output directory.

## RF-DETR Training Metrics Reports

RF-DETR training writes a sparse `metrics.csv` with training, validation, and
learning-rate rows. Generate a local training telemetry report from that file:

```bash
uv run face-benchmark report-rfdetr-training \
  --metrics-csv runs/training/<run-id>/metrics.csv \
  --run-id <run-id>
```

By default this writes under:

```text
runs/training-reports/<run-id>/
```

The report includes:

```text
metrics_clean.csv
summary.md
training_validation_loss.svg
validation_precision_recall_f1.svg
map.svg
learning_rate.svg
```

`metrics_clean.csv` merges RF-DETR's sparse rows by epoch and step, computes
validation F2 from validation precision and recall, and keeps useful numeric
training, validation, mAP, EMA mAP, and learning-rate columns. The default best
epoch/step is selected by computed validation F2 because this project prioritizes
finding faces. Use `--selection-metric f1`, `--selection-metric map50`,
`--selection-metric map50-95`, `--selection-metric precision`, or
`--selection-metric recall` to rank by another validation metric.

Compare multiple RF-DETR training reports with display labels:

```bash
uv run face-benchmark compare-rfdetr-training-runs \
  --training-run EMA1=runs/training-reports/run-a \
  --training-run EMA2=runs/training-reports/run-b \
  --run-id <comparison-run-id>
```

By default this writes under:

```text
runs/training-reports/comparisons/<comparison-run-id>/
```

Comparison artifacts include `summary.csv`, `summary.md`,
`validation_f2_overlay.svg`, `map_overlay.svg`, `loss_overlay.svg`, and
`learning_rate_overlay.svg` when learning-rate columns exist.

These training reports are not benchmark accuracy reports. They answer how a
training run evolved on RF-DETR's training and validation data. To measure a
selected checkpoint on the cleaned face dataset, use the existing benchmark or
validation workflow:

```bash
uv run face-benchmark predict-rfdetr-benchmark \
  --weights runs/training/<training-run-id>/<checkpoint>.pth \
  --run-id <validation-or-benchmark-run-id> \
  --model-name <model-name>

uv run face-benchmark validate-thresholds \
  --predictions-path runs/benchmarks/<validation-or-benchmark-run-id>/predictions/<model-name>.jsonl \
  --run-id <validation-or-benchmark-run-id>
```

Keep the artifact lanes separate:

- `runs/training-reports/` is for RF-DETR training telemetry.
- `runs/validation/` is for threshold or checkpoint validation analysis.
- `runs/benchmarks/` is for final benchmark accuracy reporting.

If `data/benchmark/target-video-test-3fps-clean/test/` is used for checkpoint
or threshold selection, treat that output as validation analysis, not as an
unbiased final test benchmark.

## Trained RF-DETR Checkpoint Evaluation

After training finishes, evaluate a trained checkpoint through the same benchmark prediction and validation commands used for any RF-DETR model. Keep the training run under `runs/training/`, then choose the checkpoint file from that run directory, such as `checkpoint_best_ema.pth` or another `.pth` file written by RF-DETR.

Generate low-threshold predictions from the trained checkpoint:

```bash
uv run face-benchmark predict-rfdetr-benchmark \
  --weights runs/training/<training-run-id>/<checkpoint>.pth \
  --run-id <model-validation-run-id> \
  --model-name <model-name>
```

If the labeled split is being used as validation data, choose an operating threshold from those predictions:

```bash
uv run face-benchmark validate-thresholds \
  --predictions-path runs/benchmarks/<model-validation-run-id>/predictions/<model-name>.jsonl \
  --run-id <model-validation-run-id> \
  --selection-metric f2
```

Compare the trained checkpoint against other validation runs:

```bash
uv run face-benchmark compare-validation-runs \
  --run-id <comparison-run-id> \
  --validation-run runs/validation/<model-validation-run-id> \
  --validation-run runs/validation/<other-validation-run-id>
```

If `data/benchmark/target-video-test-3fps-clean/` is treated as the final test benchmark, do not use it to choose thresholds, pick checkpoints, compare augmentation experiments, or make training decisions. In that case, select the checkpoint and threshold on separate validation data first, then run `evaluate-detections` once on the test benchmark with the preselected threshold.

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

## Prediction Overlay Rendering

Render annotated prediction images for one or more normalized JSONL files at selected confidence thresholds:

```bash
uv run face-benchmark render-prediction-overlays \
  --run-id model-overlay-comparison \
  --prediction-spec rfdetr-ema-1=runs/benchmarks/rfdetr-ema-1-validation/predictions/rfdetr-ema-1.jsonl:0.30 \
  --prediction-spec insightface-buffalo-l=runs/benchmarks/insightface-buffalo-l-validation/predictions/insightface-buffalo-l.jsonl:0.35 \
  --iou-threshold 0.5
```

Each `--prediction-spec` uses `label=path/to/predictions.jsonl:threshold`. The label is used as the output directory name, the path points to a normalized prediction JSONL file, and the threshold filters out lower-confidence predictions before matching.

By default this reads `data/benchmark/target-video-test-3fps-clean/test/` and writes outputs under:

```text
runs/visualizations/<run-id>/models/<label>/<image-file-name>
runs/visualizations/<run-id>/summary.csv
runs/visualizations/<run-id>/summary.json
```

When two or more specs are provided, the command also writes side-by-side comparison images under:

```text
runs/visualizations/<run-id>/comparison/<image-file-name>
```

The overlays use the same greedy matching semantics as benchmark evaluation at the selected IoU threshold. True-positive prediction boxes are green and labeled `TP <confidence>`, false-positive prediction boxes are yellow and labeled `FP <confidence>`, and unmatched ground-truth boxes are red and labeled `FN`. The summary files include TP, FP, FN, precision, recall, F1, F2, threshold, prediction path, and IoU threshold for each spec.

## Threshold Validation

Generate RF-DETR predictions directly on the cleaned COCO split before validation:

```bash
uv run face-benchmark predict-rfdetr-benchmark \
  --weights models/checkpoint_best_ema.pth \
  --run-id rfdetr-ema-1
```

This command defaults to a low inference threshold of `0.005` so validation sweeps can compare operating thresholds without being limited by an earlier `0.25` prediction filter.

### Latency Reports

Benchmark prediction commands also record inference latency. Each normalized JSONL prediction row includes `timing_ms.inference` in milliseconds. RF-DETR reports batch inference time amortized per image; InsightFace reports each image detector call directly.

By default, `predict-rfdetr-benchmark` and `predict-insightface-benchmark` write latency artifacts beside the prediction output:

```text
runs/benchmarks/<run-id>/latency/<model>.json
runs/benchmarks/<run-id>/latency.csv
```

The JSON report includes total runtime, total inference time, per-image mean, median, p90, min/max, image count, detection count, model name, backend, device, and model configuration. Latency reports are separate from accuracy evaluation; `evaluate-detections` still reports only detection metrics such as precision, recall, F scores, AP, and mAP.

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

Compare multiple validation runs with shared plots:

```bash
uv run face-benchmark compare-validation-runs \
  --run-id rfdetr-ema-comparison \
  --validation-run runs/validation/rfdetr-ema-1-validation \
  --validation-run runs/validation/rfdetr-ema-2-validation
```

This writes `summary.csv`, `summary.md`, `precision_recall_overlay.svg`, and `f1_f2_overlay.svg` under `runs/validation/comparisons/<run-id>/`. The command reads existing `threshold_validation.json` files, so it works for any model that emits normalized predictions and validation reports.

## InsightFace Benchmark Prediction

Install the optional InsightFace dependencies before running SCRFD predictions:

```bash
uv sync --extra insightface
```

The first InsightFace run may download the selected model pack into a local user cache outside this repo. Keep those cached files and generated benchmark outputs out of git. The default command uses the `buffalo_l` model pack, detector-only loading, CPU ONNX Runtime, `det_size=960`, and a low detection threshold of `0.005` so later validation sweeps can compare operating thresholds:

```bash
uv run face-benchmark predict-insightface-benchmark \
  --run-id insightface-buffalo-l-validation \
  --threshold 0.005 \
  --det-size 960
```

By default this reads `data/benchmark/target-video-test-3fps-clean/test/` and writes normalized predictions to:

```text
runs/benchmarks/insightface-buffalo-l-validation/predictions/insightface-buffalo-l.jsonl
```

Use `--providers CUDAExecutionProvider,CPUExecutionProvider` only when the local ONNX Runtime environment supports CUDA. The default `--providers CPUExecutionProvider` and `--ctx-id -1` are the portable local baseline.

Validate thresholds for InsightFace with the same command used for RF-DETR:

```bash
uv run face-benchmark validate-thresholds \
  --predictions-path runs/benchmarks/insightface-buffalo-l-validation/predictions/insightface-buffalo-l.jsonl \
  --run-id insightface-buffalo-l-validation \
  --selection-metric f2
```

If the cleaned target-domain dataset is used for this threshold selection, treat the result as validation analysis only. Do not report it later as an unbiased test-set benchmark.

Compare InsightFace against existing RF-DETR validation runs with:

```bash
uv run face-benchmark compare-validation-runs \
  --run-id model-family-comparison \
  --validation-run runs/validation/rfdetr-ema-1-validation \
  --validation-run runs/validation/rfdetr-ema-2-validation \
  --validation-run runs/validation/insightface-buffalo-l-validation
```

## EgoBlur Benchmark Prediction

Install the optional EgoBlur dependencies before running EgoBlur predictions:

```bash
uv sync --extra egoblur
```

Download the EgoBlur Gen2 face model from the Project Aria EgoBlur release
page and keep it as an ignored local artifact:

```text
models/egoblur/ego_blur_face_gen2.jit
```

The benchmark command runs EgoBlur as a detector adapter and writes normalized
face-only JSONL. EgoBlur can also detect license plates, but license plates are
not emitted into this face benchmark because the current COCO evaluation schema
only evaluates `Human face` boxes.

Generate low-threshold EgoBlur face predictions:

```bash
uv run face-benchmark predict-egoblur-benchmark \
  --run-id egoblur-gen2-face-validation \
  --threshold 0.005
```

By default the command uses EgoBlur Gen2's official resize settings,
`--resize-min 1200 --resize-max 1200`. For diagnostic comparisons with older
experiments, you can run a 704 resize variant:

```bash
uv run face-benchmark predict-egoblur-benchmark \
  --run-id egoblur-gen2-face-resize704-validation \
  --threshold 0.005 \
  --resize-min 704 \
  --resize-max 704
```

By default this reads `data/benchmark/target-video-test-3fps-clean/test/` and
writes normalized predictions to:

```text
runs/benchmarks/egoblur-gen2-face-validation/predictions/egoblur-gen2-face.jsonl
```

The command also writes latency artifacts beside the prediction output:

```text
runs/benchmarks/egoblur-gen2-face-validation/latency/egoblur-gen2-face.json
runs/benchmarks/egoblur-gen2-face-validation/latency.csv
```

EgoBlur v1 device support is `auto`, `cuda`, or `cpu`. On macOS, EgoBlur runs
on CPU and can be slow; use CUDA on Linux when available.

Validate thresholds for EgoBlur with the same workflow as RF-DETR and
InsightFace:

```bash
uv run face-benchmark validate-thresholds \
  --predictions-path runs/benchmarks/egoblur-gen2-face-validation/predictions/egoblur-gen2-face.jsonl \
  --run-id egoblur-gen2-face-validation \
  --selection-metric f2
```

Compare EgoBlur against RF-DETR and InsightFace validation runs:

```bash
uv run face-benchmark compare-validation-runs \
  --run-id face-model-comparison \
  --validation-run RF-DETR=runs/validation/<rfdetr-run> \
  --validation-run InsightFace=runs/validation/<insightface-run> \
  --validation-run EgoBlur=runs/validation/egoblur-gen2-face-validation
```

Render prediction overlays with the selected EgoBlur threshold:

```bash
uv run face-benchmark render-prediction-overlays \
  --run-id face-model-overlays \
  --prediction-spec EgoBlur=runs/benchmarks/egoblur-gen2-face-validation/predictions/egoblur-gen2-face.jsonl:<selected-threshold>
```

Do not add an EgoBlur notebook until this CLI path is working; notebooks should
remain guided wrappers around the CLI source of truth.

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
