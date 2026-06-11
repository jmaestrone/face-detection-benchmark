# Dataset Policy

This project has four separate data lanes. Keep them distinct in naming, storage, review status, and interpretation of results.

## Data Lanes

### Training Data

Training data lives under `data/training/`.

Use it for model training, fine-tuning, augmentation experiments, and RF-DETR training/validation splits. The default RF-DETR training root is:

```text
data/training/rfdetr/
```

Training outputs belong under:

```text
runs/training/
```

The `train-rfdetr` command requires an explicit `--dataset-dir` and rejects paths under `data/benchmark/`.

### Validation Data

Validation data is labeled data used to choose thresholds, checkpoints, model families, resize settings, or other operating decisions.

Validation analysis outputs belong under:

```text
runs/validation/
runs/validation/comparisons/
```

If the cleaned target-domain dataset is used for threshold selection, checkpoint selection, or model comparison, that use is validation analysis. Do not later present the same dataset/run as an unbiased final test benchmark.

### Final Test Benchmark Data

The canonical cleaned target-domain final test benchmark is the reviewed Roboflow dataset downloaded to:

```text
data/benchmark/target-video-test-3fps-clean/test/
```

Treat it as final test data only when thresholds, checkpoints, and model variants were selected elsewhere first.

Final benchmark evaluation should be a one-shot measurement with a preselected threshold:

```bash
uv run face-benchmark evaluate-detections \
  --predictions-path runs/benchmarks/<run-id>/predictions/<model>.jsonl \
  --confidence-threshold <preselected-threshold>
```

Do not train, fine-tune, tune thresholds, pick checkpoints, compare augmentation experiments, or repeatedly sweep decisions on final test data.

### Pre-label Batches

Pre-label batches are temporary sampled frames plus RF-DETR detections prepared for Roboflow manual review.

Typical local paths:

```text
data/frames/
data/predictions/
data/roboflow-export/
runs/previews/
runs/video-summaries/
```

Pre-label batches are not training data and are not final benchmark data until they have been manually reviewed, versioned in Roboflow, downloaded, and validated as the intended dataset lane.

## Code-backed Checks

The current implementation enforces several policy constraints:

- `train-rfdetr` refuses dataset paths under `data/benchmark/`.
- `download-roboflow-benchmark` validates a test-only Roboflow export by checking the expected split, category, image count, and empty non-test splits.
- The default cleaned benchmark dataset name is `target-video-test-3fps-clean`.
- The expected face category is `Human face`.
- The current expected cleaned benchmark image count is `169`.

## Artifact Lanes

Keep generated outputs under ignored local paths:

```text
data/benchmark/           # downloaded benchmark or validation datasets
data/training/            # training datasets
runs/benchmarks/          # final benchmark predictions, metrics, leaderboards
runs/validation/          # threshold and checkpoint validation
runs/training/            # RF-DETR training outputs and checkpoints
runs/training-reports/    # RF-DETR training telemetry reports
runs/visualizations/      # prediction overlays
```

`runs/benchmarks/` can contain low-threshold prediction files used before validation. The interpretation depends on how the labeled dataset is used: validation decisions belong under `runs/validation/`; final accuracy claims belong under `runs/benchmarks/` only after thresholds and checkpoints were selected elsewhere.
