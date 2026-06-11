# RF-DETR Training

Use this guide for RF-DETR training datasets, training runs, telemetry reports, and trained checkpoint validation/evaluation.

## Dataset Rules

RF-DETR training must use an explicit training dataset directory. Keep training datasets under:

```text
data/training/
data/training/rfdetr/
```

Keep training outputs under:

```text
runs/training/
```

Never point `train-rfdetr` at `data/benchmark/`. The command validates this and fails before loading the model if `--dataset-dir` is inside `data/benchmark/`.

## Training

Run training from an explicit dataset directory:

```bash
uv run face-benchmark train-rfdetr \
  --dataset-dir data/training/rfdetr \
  --output-dir runs/training/<run-id> \
  --epochs 100 \
  --batch-size 4 \
  --device auto
```

By default the command uses RF-DETR's `roboflow` dataset format. Use `--dataset-file coco`, `--dataset-file yolo`, or `--dataset-file o365` only when the training dataset matches that RF-DETR format.

Use `--weights models/<checkpoint>.pth` when fine-tuning from a local checkpoint.

Training dependencies are optional in RF-DETR and may not be present in a default local environment. If a training run reports missing RF-DETR training dependencies, install the RF-DETR train extras in the local environment before running a real training job.

## Training Outputs

The command writes reproducibility artifacts before training starts:

```text
runs/training/<run-id>/config.json
runs/training/<run-id>/metadata.json
```

RF-DETR writes its own training outputs and checkpoints under the same output directory.

## Training Telemetry Reports

RF-DETR training writes a sparse `metrics.csv` with training, validation, and learning-rate rows. Generate a local telemetry report:

```bash
uv run face-benchmark report-rfdetr-training \
  --metrics-csv runs/training/<run-id>/metrics.csv \
  --run-id <run-id>
```

By default this writes under:

```text
runs/training-reports/<run-id>/
```

Report artifacts include:

```text
metrics_clean.csv
metrics.md
summary.md
training_validation_loss.svg
validation_precision_recall_f1.svg
map.svg
learning_rate.svg
```

`metrics_clean.csv` merges RF-DETR's sparse rows by epoch and step. `metrics.md` writes the same cleaned rows as a Markdown table. Both compute validation F2 from validation precision and recall and keep useful numeric training, validation, mAP, EMA mAP, and learning-rate columns.

The default best epoch/step is selected by computed validation F2 because this project prioritizes finding faces. Use `--selection-metric f1`, `--selection-metric map50`, `--selection-metric map50-95`, `--selection-metric precision`, or `--selection-metric recall` to rank by another validation metric.

Compare multiple RF-DETR training reports:

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

Comparison artifacts include `summary.csv`, `summary.md`, `validation_f2_overlay.svg`, `map_overlay.svg`, `loss_overlay.svg`, and `learning_rate_overlay.svg` when learning-rate columns exist.

Training telemetry reports are not benchmark accuracy reports. They answer how a training run evolved on RF-DETR's training and validation data.

## Checkpoint Validation And Evaluation

After training, evaluate a selected checkpoint through the same prediction, validation, and benchmark commands used for any RF-DETR model.

Generate low-threshold predictions:

```bash
uv run face-benchmark predict-rfdetr-benchmark \
  --weights runs/training/<training-run-id>/<checkpoint>.pth \
  --run-id <model-validation-run-id> \
  --model-name <model-name>
```

If the labeled split is validation data, choose an operating threshold:

```bash
uv run face-benchmark validate-thresholds \
  --predictions-path runs/benchmarks/<model-validation-run-id>/predictions/<model-name>.jsonl \
  --run-id <model-validation-run-id> \
  --selection-metric f2
```

Compare trained checkpoints against other validation runs:

```bash
uv run face-benchmark compare-validation-runs \
  --run-id <comparison-run-id> \
  --validation-run runs/validation/<model-validation-run-id> \
  --validation-run runs/validation/<other-validation-run-id>
```

If `data/benchmark/target-video-test-3fps-clean/` is treated as final test data, do not use it to choose thresholds, pick checkpoints, compare augmentation experiments, or make training decisions. Select the checkpoint and threshold on separate validation data first, then run `evaluate-detections` once on the test benchmark with the preselected threshold.
