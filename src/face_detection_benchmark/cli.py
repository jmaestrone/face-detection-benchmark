"""Typer app registration for the face detection benchmark pipeline."""

from __future__ import annotations

import typer

from face_detection_benchmark.commands.datasets import (
    download_roboflow_benchmark,
    export_coco,
    upload_roboflow,
)
from face_detection_benchmark.commands.evaluation import (
    compare_validation_runs,
    evaluate_detections,
    validate_thresholds,
)
from face_detection_benchmark.commands.insightface import predict_insightface_benchmark
from face_detection_benchmark.commands.rfdetr import (
    predict_faces,
    predict_rfdetr_benchmark,
)
from face_detection_benchmark.commands.video import extract_frames

app = typer.Typer(
    help="Extract frames, run face detection models, and export benchmark datasets."
)


def register_commands(typer_app: typer.Typer) -> None:
    """Register all command implementations on the Typer app."""
    typer_app.command()(extract_frames)
    typer_app.command()(predict_faces)
    typer_app.command()(predict_rfdetr_benchmark)
    typer_app.command()(predict_insightface_benchmark)
    typer_app.command()(export_coco)
    typer_app.command()(download_roboflow_benchmark)
    typer_app.command()(evaluate_detections)
    typer_app.command()(validate_thresholds)
    typer_app.command()(compare_validation_runs)
    typer_app.command()(upload_roboflow)


register_commands(app)
