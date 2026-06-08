"""Video-related CLI command implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from face_detection_benchmark.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_FPS,
    DEFAULT_FRAMES_DIR,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_RUNS_DIR,
    DEFAULT_VIDEO_DIR,
)
from face_detection_benchmark.commands.common import default_run_id
from face_detection_benchmark.reports import (
    summarize_video_predictions as write_video_prediction_summaries,
)
from face_detection_benchmark.video import METADATA_FILE_NAME
from face_detection_benchmark.video import extract_video_frames


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


def summarize_video_predictions(
    predictions_path: Annotated[
        Path,
        typer.Option(
            "--predictions-path",
            help="Extracted-frame prediction JSONL to summarize.",
        ),
    ] = DEFAULT_PREDICTIONS_PATH,
    metadata_path: Annotated[
        Path,
        typer.Option(
            "--metadata-path",
            help="Frame extraction metadata JSONL path.",
        ),
    ] = DEFAULT_FRAMES_DIR / METADATA_FILE_NAME,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help=(
                "Video summary output directory. Defaults to "
                "runs/video-summaries/<run-id>."
            ),
        ),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(
            "--run-id",
            help="Run id used when --output-dir is not supplied.",
        ),
    ] = None,
    confidence_threshold: Annotated[
        float,
        typer.Option(
            "--confidence-threshold",
            min=0.0,
            max=1.0,
            help="Only include detections at or above this confidence.",
        ),
    ] = DEFAULT_CONFIDENCE_THRESHOLD,
) -> None:
    """Summarize extracted-frame predictions by source video without accuracy claims."""
    try:
        resolved_run_id = run_id or default_run_id()
        resolved_output_dir = output_dir or (
            DEFAULT_RUNS_DIR / "video-summaries" / resolved_run_id
        )
        report_paths = write_video_prediction_summaries(
            predictions_path=predictions_path,
            metadata_path=metadata_path,
            output_dir=resolved_output_dir,
            confidence_threshold=confidence_threshold,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    typer.echo(f"Video summaries: {resolved_output_dir}")
    typer.echo(f"Summary JSON: {report_paths['summary_json_path']}")
    typer.echo(f"Summary CSV: {report_paths['summary_csv_path']}")
