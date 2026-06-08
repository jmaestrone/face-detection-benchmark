"""Summarize extracted-frame predictions by source video."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from face_detection_benchmark.video import probe_video


def summarize_video_predictions(
    *,
    predictions_path: Path,
    metadata_path: Path,
    output_dir: Path,
    confidence_threshold: float,
) -> dict[str, Path]:
    """Write per-video and aggregate summaries for extracted-frame predictions."""
    _validate_inputs(predictions_path, metadata_path, confidence_threshold)
    metadata_by_key = _read_metadata_by_key(metadata_path)
    prediction_rows = _read_jsonl(predictions_path, "prediction")
    if not prediction_rows:
        raise ValueError(f"No prediction rows found in {predictions_path}")

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for prediction_row in prediction_rows:
        metadata_row = _metadata_for_prediction(prediction_row, metadata_by_key)
        source_video = (
            prediction_row.get("source_video") or metadata_row.get("source_video") or ""
        )
        video_stem = (
            prediction_row.get("video_stem")
            or metadata_row.get("video_stem")
            or _video_stem_from_source(source_video)
        )
        if not source_video or not video_stem:
            raise ValueError(
                "Prediction rows must include source_video/video_stem or match "
                "metadata by file_name/image_path"
            )
        normalized_row = {
            **prediction_row,
            "source_video": source_video,
            "video_stem": video_stem,
            "frame_index": prediction_row.get(
                "frame_index", metadata_row.get("frame_index")
            ),
            "timestamp_seconds": prediction_row.get(
                "timestamp_seconds", metadata_row.get("timestamp_seconds")
            ),
            "metadata": metadata_row,
        }
        grouped_rows[(source_video, video_stem)].append(normalized_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    video_summaries = [
        _summarize_group(
            source_video=source_video,
            video_stem=video_stem,
            rows=rows,
            confidence_threshold=confidence_threshold,
        )
        for (source_video, video_stem), rows in sorted(
            grouped_rows.items(), key=lambda item: item[0][1]
        )
    ]

    for video_summary in video_summaries:
        summary_path = output_dir / video_summary["video_stem"] / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(video_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    aggregate_summary = _aggregate_summary(
        video_summaries=video_summaries,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        confidence_threshold=confidence_threshold,
    )
    summary_json_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path.write_text(
        json.dumps(aggregate_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(video_summaries, summary_csv_path)
    return {
        "summary_json_path": summary_json_path,
        "summary_csv_path": summary_csv_path,
    }


def _validate_inputs(
    predictions_path: Path,
    metadata_path: Path,
    confidence_threshold: float,
) -> None:
    if not predictions_path.exists():
        raise ValueError(f"Predictions file does not exist: {predictions_path}")
    if not metadata_path.exists():
        raise ValueError(f"Frame metadata file does not exist: {metadata_path}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be between 0 and 1")


def _read_jsonl(path: Path, row_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{row_name.title()} row {line_number} in {path} is not an object"
                )
            rows.append(payload)
    return rows


def _read_metadata_by_key(metadata_path: Path) -> dict[str, dict[str, Any]]:
    metadata_rows = _read_jsonl(metadata_path, "metadata")
    if not metadata_rows:
        raise ValueError(f"No frame metadata rows found in {metadata_path}")

    metadata_by_key: dict[str, dict[str, Any]] = {}
    for metadata_row in metadata_rows:
        for key in _metadata_keys(metadata_row):
            metadata_by_key.setdefault(key, metadata_row)
    return metadata_by_key


def _metadata_keys(row: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (
            row.get("file_name"),
            row.get("output_path"),
            row.get("image_path"),
        )
        if value
    }


def _metadata_for_prediction(
    prediction_row: dict[str, Any],
    metadata_by_key: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for value in (
        prediction_row.get("file_name"),
        prediction_row.get("image_path"),
        Path(str(prediction_row.get("image_path"))).name
        if prediction_row.get("image_path")
        else None,
    ):
        if value and str(value) in metadata_by_key:
            return metadata_by_key[str(value)]
    return {}


def _summarize_group(
    *,
    source_video: str,
    video_stem: str,
    rows: list[dict[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            _sortable_float(row.get("timestamp_seconds")),
            _sortable_int(row.get("frame_index")),
            str(row.get("file_name", "")),
        ),
    )
    model_names = sorted(
        {str(row.get("model_name")) for row in sorted_rows if row.get("model_name")}
    )
    filtered_frame_rows = [
        _frame_summary(row, confidence_threshold) for row in sorted_rows
    ]
    frames_with_faces = [
        frame_row for frame_row in filtered_frame_rows if frame_row["detection_count"]
    ]
    timing_values = [
        timing_value
        for row in sorted_rows
        for timing_value in _timing_values_ms(row.get("timing_ms"))
    ]
    timestamps = [
        float(row["timestamp_seconds"])
        for row in sorted_rows
        if row.get("timestamp_seconds") is not None
    ]

    return {
        "source_video": source_video,
        "video_stem": video_stem,
        "model_name": _single_or_list(model_names),
        "confidence_threshold": confidence_threshold,
        "processed_frame_count": len(sorted_rows),
        "frames_with_faces": len(frames_with_faces),
        "total_detections": sum(row["detection_count"] for row in filtered_frame_rows),
        "timestamps_with_faces": [
            row["timestamp_seconds"]
            for row in frames_with_faces
            if row["timestamp_seconds"] is not None
        ],
        "detections_by_timestamp": frames_with_faces,
        "sampled_fps": _sampled_fps(timestamps),
        "source_fps": _source_fps(source_video, sorted_rows),
        "latency": _latency_summary(
            processed_frame_count=len(sorted_rows),
            timing_values_ms=timing_values,
        ),
    }


def _frame_summary(
    row: dict[str, Any],
    confidence_threshold: float,
) -> dict[str, Any]:
    detections = [
        detection
        for detection in row.get("detections", [])
        if _passes_threshold(detection, confidence_threshold)
    ]
    return {
        "timestamp_seconds": row.get("timestamp_seconds"),
        "frame_index": row.get("frame_index"),
        "file_name": row.get("file_name"),
        "detection_count": len(detections),
        "detections": detections,
    }


def _passes_threshold(detection: dict[str, Any], confidence_threshold: float) -> bool:
    confidence = detection.get("confidence")
    if confidence is None:
        return False
    return float(confidence) >= confidence_threshold


def _timing_values_ms(raw_timing: Any) -> list[float]:
    if not isinstance(raw_timing, dict):
        return []
    timing_values: list[float] = []
    for value in raw_timing.values():
        if isinstance(value, int | float):
            timing_values.append(float(value))
    return timing_values


def _latency_summary(
    *,
    processed_frame_count: int,
    timing_values_ms: list[float],
) -> dict[str, float | int | None]:
    if not timing_values_ms:
        return {
            "timed_frame_count": 0,
            "total_ms": None,
            "mean_ms": None,
            "min_ms": None,
            "max_ms": None,
            "processed_fps": None,
        }
    total_ms = sum(timing_values_ms)
    return {
        "timed_frame_count": len(timing_values_ms),
        "total_ms": round(total_ms, 4),
        "mean_ms": round(total_ms / len(timing_values_ms), 4),
        "min_ms": round(min(timing_values_ms), 4),
        "max_ms": round(max(timing_values_ms), 4),
        "processed_fps": round(processed_frame_count / (total_ms / 1000), 4)
        if total_ms > 0
        else None,
    }


def _sampled_fps(timestamps: list[float]) -> float | None:
    unique_timestamps = sorted(set(timestamps))
    if len(unique_timestamps) < 2:
        return None
    span_seconds = unique_timestamps[-1] - unique_timestamps[0]
    if span_seconds <= 0:
        return None
    return round((len(unique_timestamps) - 1) / span_seconds, 4)


def _source_fps(source_video: str, rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        metadata = row.get("metadata", {})
        for key in ("source_fps", "native_fps", "fps"):
            if isinstance(metadata, dict) and metadata.get(key) is not None:
                return round(float(metadata[key]), 4)
    source_path = Path(source_video)
    if source_path.exists():
        try:
            return round(probe_video(source_path).fps, 4)
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None
    return None


def _aggregate_summary(
    *,
    video_summaries: list[dict[str, Any]],
    predictions_path: Path,
    metadata_path: Path,
    confidence_threshold: float,
) -> dict[str, Any]:
    return {
        "predictions_path": predictions_path.as_posix(),
        "metadata_path": metadata_path.as_posix(),
        "confidence_threshold": confidence_threshold,
        "video_count": len(video_summaries),
        "processed_frame_count": sum(
            summary["processed_frame_count"] for summary in video_summaries
        ),
        "frames_with_faces": sum(
            summary["frames_with_faces"] for summary in video_summaries
        ),
        "total_detections": sum(
            summary["total_detections"] for summary in video_summaries
        ),
        "videos": video_summaries,
    }


def _write_summary_csv(
    video_summaries: list[dict[str, Any]],
    summary_csv_path: Path,
) -> None:
    fieldnames = [
        "source_video",
        "video_stem",
        "model_name",
        "confidence_threshold",
        "processed_frame_count",
        "frames_with_faces",
        "total_detections",
        "sampled_fps",
        "source_fps",
        "timed_frame_count",
        "total_latency_ms",
        "mean_latency_ms",
        "processed_fps",
    ]
    with summary_csv_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in video_summaries:
            latency = summary["latency"]
            writer.writerow(
                {
                    "source_video": summary["source_video"],
                    "video_stem": summary["video_stem"],
                    "model_name": summary["model_name"],
                    "confidence_threshold": summary["confidence_threshold"],
                    "processed_frame_count": summary["processed_frame_count"],
                    "frames_with_faces": summary["frames_with_faces"],
                    "total_detections": summary["total_detections"],
                    "sampled_fps": summary["sampled_fps"],
                    "source_fps": summary["source_fps"],
                    "timed_frame_count": latency["timed_frame_count"],
                    "total_latency_ms": latency["total_ms"],
                    "mean_latency_ms": latency["mean_ms"],
                    "processed_fps": latency["processed_fps"],
                }
            )


def _video_stem_from_source(source_video: str) -> str:
    return Path(source_video).stem if source_video else ""


def _single_or_list(values: list[str]) -> str | list[str]:
    if len(values) == 1:
        return values[0]
    return values


def _sortable_float(value: Any) -> float:
    return float(value) if value is not None else float("inf")


def _sortable_int(value: Any) -> int:
    return int(value) if value is not None else 0
