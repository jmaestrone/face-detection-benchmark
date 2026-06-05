"""InsightFace prediction CLI command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from face_detection_benchmark.commands.common import default_run_id
from face_detection_benchmark.config import (
    DEFAULT_BENCHMARK_DATA_DIR,
    DEFAULT_BENCHMARK_DATASET_NAME,
    DEFAULT_ROBOFLOW_TEST_SPLIT,
    DEFAULT_RUNS_DIR,
)
from face_detection_benchmark.inference import predict_insightface_from_coco_dataset
from face_detection_benchmark.models.insightface import (
    DEFAULT_INSIGHTFACE_CTX_ID,
    DEFAULT_INSIGHTFACE_DET_SIZE,
    DEFAULT_INSIGHTFACE_MODEL_NAME,
    DEFAULT_INSIGHTFACE_MODEL_PACK,
    DEFAULT_INSIGHTFACE_PROVIDERS,
    DEFAULT_INSIGHTFACE_THRESHOLD,
    parse_providers,
)


def predict_insightface_benchmark(
    dataset_dir: Annotated[
        Path,
        typer.Option(
            "--dataset-dir",
            help="COCO benchmark split directory containing _annotations.coco.json.",
        ),
    ] = DEFAULT_BENCHMARK_DATA_DIR
    / DEFAULT_BENCHMARK_DATASET_NAME
    / DEFAULT_ROBOFLOW_TEST_SPLIT,
    output_path: Annotated[
        Path | None,
        typer.Option(
            "--output-path",
            "-o",
            help=(
                "JSONL output path. Defaults to "
                "runs/benchmarks/<run-id>/predictions/<model>.jsonl."
            ),
        ),
    ] = None,
    model_name: Annotated[
        str,
        typer.Option(
            "--model-name",
            help="Model name written into prediction rows.",
        ),
    ] = DEFAULT_INSIGHTFACE_MODEL_NAME,
    model_pack: Annotated[
        str,
        typer.Option(
            "--model-pack",
            help="InsightFace model pack name.",
        ),
    ] = DEFAULT_INSIGHTFACE_MODEL_PACK,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Run id used when --output-path is not supplied.",
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            max=1.0,
            help="Low InsightFace detection threshold used before validation sweeps.",
        ),
    ] = DEFAULT_INSIGHTFACE_THRESHOLD,
    det_size: Annotated[
        int,
        typer.Option(
            "--det-size",
            min=1,
            help="Square InsightFace detector input size.",
        ),
    ] = DEFAULT_INSIGHTFACE_DET_SIZE,
    providers: Annotated[
        str,
        typer.Option(
            "--providers",
            help="Comma-separated ONNX Runtime providers.",
        ),
    ] = ",".join(DEFAULT_INSIGHTFACE_PROVIDERS),
    ctx_id: Annotated[
        int,
        typer.Option(
            "--ctx-id",
            help="InsightFace context id. Use -1 for CPU.",
        ),
    ] = DEFAULT_INSIGHTFACE_CTX_ID,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Number of images to predict per batch.",
        ),
    ] = 4,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Only process the first N images for a smoke run.",
        ),
    ] = None,
    preview_dir: Annotated[
        Path | None,
        typer.Option(
            "--preview-dir",
            help="Optional directory for annotated preview images.",
        ),
    ] = None,
    max_previews: Annotated[
        int,
        typer.Option(
            "--max-previews",
            min=0,
            help="Maximum number of preview images to write when --preview-dir is set.",
        ),
    ] = 20,
) -> None:
    """Run InsightFace/SCRFD on the local COCO benchmark split."""
    try:
        resolved_run_id = run_id or default_run_id()
        resolved_output_path = output_path or (
            DEFAULT_RUNS_DIR
            / "benchmarks"
            / resolved_run_id
            / "predictions"
            / f"{model_name}.jsonl"
        )
        result = predict_insightface_from_coco_dataset(
            dataset_dir=dataset_dir,
            output_path=resolved_output_path,
            model_name=model_name,
            model_pack=model_pack,
            threshold=threshold,
            det_size=det_size,
            providers=parse_providers(providers),
            ctx_id=ctx_id,
            batch_size=batch_size,
            limit=limit,
            preview_dir=preview_dir,
            max_previews=max_previews,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Wrote detections for {result.image_count} benchmark images "
        f"({result.detection_count} boxes) to {result.output_path}"
    )
    if result.preview_count:
        typer.echo(f"Preview images: {result.preview_count} in {result.preview_dir}")
