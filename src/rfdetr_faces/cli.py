"""Command-line entrypoints for the RF-DETR faces pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rfdetr_faces.config import DEFAULT_FRAME_FPS, DEFAULT_FRAMES_DIR, DEFAULT_VIDEO_DIR
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
def predict_faces() -> None:
    """Run RF-DETR face detection on extracted frames."""
    typer.echo("predict-faces is planned for a later checkpoint.")


@app.command()
def export_coco() -> None:
    """Export RF-DETR detections as COCO ground-truth annotations."""
    typer.echo("export-coco is planned for a later checkpoint.")


@app.command()
def upload_roboflow() -> None:
    """Upload the exported COCO dataset to Roboflow."""
    typer.echo("upload-roboflow is planned for a later checkpoint.")
