"""Command-line entrypoints for the RF-DETR faces pipeline."""

from __future__ import annotations

import typer

app = typer.Typer(
    help="Extract frames, run RF-DETR face detection, and export Roboflow datasets."
)


@app.command()
def extract_frames() -> None:
    """Extract sampled frames from the source videos."""
    typer.echo("extract-frames is planned for the next checkpoint.")


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
