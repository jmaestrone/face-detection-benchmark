# Evaluation Workflows

Use this guide for model prediction, threshold validation, final benchmark evaluation, latency reports, and prediction overlays.

## Prediction Outputs

Benchmark prediction commands read a COCO split directory and write normalized JSONL predictions under:

```text
runs/benchmarks/<run-id>/predictions/<model>.jsonl
```

The default dataset path is:

```text
data/benchmark/target-video-test-3fps-clean/test/
```

Prediction rows must use `file_name` values that match the COCO image filenames.

## RF-DETR Prediction

Generate low-threshold RF-DETR predictions:

```bash
uv run face-benchmark predict-rfdetr-benchmark \
  --weights models/checkpoint_best_ema.pth \
  --run-id rfdetr-ema-1
```

The command defaults to a low inference threshold of `0.005` so later validation sweeps can compare operating thresholds without being limited by an earlier prediction filter.

## InsightFace Prediction

Install optional dependencies first:

```bash
uv sync --extra insightface
```

Run the portable local baseline:

```bash
uv run face-benchmark predict-insightface-benchmark \
  --run-id insightface-buffalo-l-validation \
  --threshold 0.005 \
  --det-size 960
```

The default model pack is `buffalo_l`, the default model name is `insightface-buffalo-l`, and the default ONNX Runtime provider is CPU.

## EgoBlur Prediction

Install optional dependencies first:

```bash
uv sync --extra egoblur
```

Keep the EgoBlur Gen2 face TorchScript model at:

```text
models/egoblur/ego_blur_face_gen2.jit
```

Run EgoBlur face prediction:

```bash
uv run face-benchmark predict-egoblur-benchmark \
  --run-id egoblur-gen2-face-validation \
  --threshold 0.005
```

By default the command uses EgoBlur Gen2 resize settings:

```text
--resize-min 1200 --resize-max 1200
```

For diagnostic comparisons with older experiments, a 704 resize variant can be run explicitly:

```bash
uv run face-benchmark predict-egoblur-benchmark \
  --run-id egoblur-gen2-face-resize704-validation \
  --threshold 0.005 \
  --resize-min 704 \
  --resize-max 704
```

## Threshold Validation

Use threshold validation only on data that is intentionally being treated as validation data:

```bash
uv run face-benchmark validate-thresholds \
  --predictions-path runs/benchmarks/<run-id>/predictions/<model>.jsonl \
  --run-id <run-id> \
  --selection-metric f2
```

By default this reads the same COCO split path as benchmark evaluation, evaluates thresholds `0.005`, `0.01`, and `0.05` through `0.80` in `0.05` steps at IoU `0.50`, and writes:

```text
runs/validation/<run-id>/threshold_validation.json
runs/validation/<run-id>/selected_threshold.json
runs/validation/<run-id>/threshold_metrics.csv
runs/validation/<run-id>/threshold_metrics.md
runs/validation/<run-id>/precision_recall.svg
runs/validation/<run-id>/f1_f2_by_threshold.svg
```

Use `--selection-metric f1`, `--selection-metric f2`, `--selection-metric precision`, or `--selection-metric recall`. If multiple thresholds tie on the selected metric, the higher threshold is selected. Use `--thresholds 0.10,0.20,0.30` to provide a custom threshold grid.

If the cleaned target-domain dataset is used for threshold selection, treat the result as validation analysis only.

Compare validation runs:

```bash
uv run face-benchmark compare-validation-runs \
  --run-id model-family-comparison \
  --validation-run RF-DETR=runs/validation/<rfdetr-run> \
  --validation-run InsightFace=runs/validation/<insightface-run> \
  --validation-run EgoBlur=runs/validation/<egoblur-run>
```

This writes under:

```text
runs/validation/comparisons/<run-id>/
```

## Final Benchmark Evaluation

Use final benchmark evaluation only after threshold, checkpoint, and model selection decisions have already been made on separate validation data:

```bash
uv run face-benchmark evaluate-detections \
  --predictions-path runs/benchmarks/<run-id>/predictions/<model>.jsonl \
  --confidence-threshold <preselected-threshold>
```

By default this reads:

```text
data/benchmark/target-video-test-3fps-clean/test/
```

It writes metrics under `runs/benchmarks/<new-run-id>/`, appends one row to:

```text
runs/benchmarks/results.csv
```

and regenerates:

```text
runs/benchmarks/results.md
```

The evaluator reports precision, recall, F1, F2 at IoU 0.50, `AP50`, `AP75`, and `mAP@[0.50:0.95]`.

Choose `--confidence-threshold` before evaluating final test data. The diagnostic `--include-confidence-sweep` option should not be used for threshold selection on a final test benchmark.

## Latency Reports

Benchmark prediction commands record inference latency beside prediction outputs:

```text
runs/benchmarks/<run-id>/latency/<model>.json
runs/benchmarks/<run-id>/latency.csv
```

Each normalized JSONL prediction row includes `timing_ms.inference` in milliseconds. RF-DETR reports batch inference time amortized per image; InsightFace and EgoBlur report detector runtime through their adapters.

Latency reports are separate from accuracy evaluation.

## Prediction Overlays

Render TP/FP/FN overlays for one or more prediction files:

```bash
uv run face-benchmark render-prediction-overlays \
  --run-id model-overlay-comparison \
  --prediction-spec RF-DETR=runs/benchmarks/<rfdetr-run>/predictions/<model>.jsonl:<threshold> \
  --prediction-spec InsightFace=runs/benchmarks/<insightface-run>/predictions/<model>.jsonl:<threshold> \
  --iou-threshold 0.5
```

Each `--prediction-spec` uses:

```text
label=path/to/predictions.jsonl:threshold
```

By default overlays are written under:

```text
runs/visualizations/<run-id>/models/<label>/<image-file-name>
runs/visualizations/<run-id>/summary.csv
runs/visualizations/<run-id>/summary.json
```

When two or more specs are provided, side-by-side comparisons are also written:

```text
runs/visualizations/<run-id>/comparison/<image-file-name>
```
