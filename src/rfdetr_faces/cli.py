"""Command-line entrypoints for the RF-DETR faces pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rfdetr_faces.coco import export_predictions_to_coco
from rfdetr_faces.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_FPS,
    DEFAULT_FRAMES_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_ROBOFLOW_EXPORT_DIR,
    DEFAULT_VIDEO_DIR,
)
from rfdetr_faces.inference import predict_faces_from_frames
from rfdetr_faces.video import extract_video_frames

app = typer.Typer(
    help="Extract frames, run RF-DETR face detection, and export Roboflow datasets."
)


@app.command()
def extract_frames(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory containing source MP4 videos.",
        ),
    ] = DEFAULT_VIDEO_DIR,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where extracted frames and metadata are written.",
        ),
    ] = DEFAULT_FRAMES_DIR,
    fps: Annotated[
        float,
        typer.Option(
            "--fps",
            min=0.001,
            help="Sample this many frames per second.",
        ),
    ] = DEFAULT_FRAME_FPS,
    every_n_frames: Annotated[
        int | None,
        typer.Option(
            "--every-n-frames",
            min=1,
            help="Sample by source frame interval instead of seconds.",
        ),
    ] = None,
    image_format: Annotated[
        str,
        typer.Option(
            "--image-format",
            help="Image format for extracted frames: jpg, jpeg, or png.",
        ),
    ] = "jpg",
    quality: Annotated[
        int,
        typer.Option(
            "--quality",
            min=1,
            max=100,
            help="JPEG quality when writing jpg/jpeg frames.",
        ),
    ] = 95,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite existing extracted frame images.",
        ),
    ] = False,
) -> None:
    """Extract sampled frames from the source videos."""
    try:
        result = extract_video_frames(
            input_dir=input_dir,
            output_dir=output_dir,
            fps=fps,
            every_n_frames=every_n_frames,
            image_format=image_format,
            quality=quality,
            overwrite=overwrite,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Extracted {result.frame_count} frames from {result.video_count} videos "
        f"to {result.output_dir}"
    )
    typer.echo(f"Metadata: {result.metadata_path}")


@app.command()
def predict_faces(
    frames_dir: Annotated[
        Path,
        typer.Option(
            "--frames-dir",
            help="Directory containing extracted frames and metadata.jsonl.",
        ),
    ] = DEFAULT_FRAMES_DIR,
    metadata_path: Annotated[
        Path | None,
        typer.Option(
            "--metadata-path",
            help="Frame metadata JSONL path. Defaults to <frames-dir>/metadata.jsonl.",
        ),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option(
            "--output-path",
            "-o",
            help="JSONL path where RF-DETR detections are written.",
        ),
    ] = DEFAULT_PREDICTIONS_PATH,
    weights: Annotated[
        Path,
        typer.Option(
            "--weights",
            help="Local RF-DETR checkpoint path.",
        ),
    ] = DEFAULT_MODEL_PATH,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            max=1.0,
            help="Confidence threshold for RF-DETR detections.",
        ),
    ] = DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Number of frames to predict per RF-DETR batch.",
        ),
    ] = 4,
    max_detections: Annotated[
        int,
        typer.Option(
            "--max-detections",
            min=1,
            help="Maximum face detections requested from RF-DETR per frame.",
        ),
    ] = 40,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help="Device to use: auto, mps, cuda, or cpu.",
        ),
    ] = "auto",
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Only process the first N frames for a smoke run.",
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
    """Run RF-DETR face detection on extracted frames."""
    try:
        result = predict_faces_from_frames(
            frames_dir=frames_dir,
            metadata_path=metadata_path,
            output_path=output_path,
            weights_path=weights,
            threshold=threshold,
            batch_size=batch_size,
            max_detections=max_detections,
            device=device,
            limit=limit,
            preview_dir=preview_dir,
            max_previews=max_previews,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Wrote detections for {result.image_count} frames "
        f"({result.detection_count} boxes) to {result.output_path}"
    )
    if result.preview_count:
        typer.echo(f"Preview images: {result.preview_count} in {result.preview_dir}")


@app.command()
def export_coco(
    frames_dir: Annotated[
        Path,
        typer.Option(
            "--frames-dir",
            help="Directory containing extracted frames and metadata.jsonl.",
        ),
    ] = DEFAULT_FRAMES_DIR,
    predictions_path: Annotated[
        Path,
        typer.Option(
            "--predictions-path",
            help="JSONL predictions path written by predict-faces.",
        ),
    ] = DEFAULT_PREDICTIONS_PATH,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory where the Roboflow-ready COCO export is written.",
        ),
    ] = DEFAULT_ROBOFLOW_EXPORT_DIR,
    include_empty: Annotated[
        bool,
        typer.Option(
            "--include-empty",
            help="Include frames with no detections in the exported dataset.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite image files already present in the export directory.",
        ),
    ] = False,
) -> None:
    """Export RF-DETR detections as COCO ground-truth annotations."""
    try:
        result = export_predictions_to_coco(
            frames_dir=frames_dir,
            predictions_path=predictions_path,
            output_dir=output_dir,
            include_empty=include_empty,
            overwrite=overwrite,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(
        f"Exported {result.image_count} images and {result.annotation_count} "
        f"annotations to {result.dataset_dir}"
    )
    typer.echo(f"COCO annotations: {result.annotations_path}")
    if result.clipped_box_count:
        typer.echo(f"Clipped boxes at image bounds: {result.clipped_box_count}")
    if result.skipped_box_count:
        typer.echo(f"Skipped invalid boxes: {result.skipped_box_count}")


@app.command()
def upload_roboflow() -> None:
    """Upload the exported COCO dataset to Roboflow."""
    typer.echo("upload-roboflow is planned for a later checkpoint.")
