"""Video frame extraction helpers."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable

METADATA_FILE_NAME = "metadata.jsonl"
SUPPORTED_VIDEO_EXTENSIONS = {".mp4"}
SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png"}


@dataclass(frozen=True)
class VideoInfo:
    """Metadata needed to sample frames from a video."""

    path: Path
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int | None


@dataclass(frozen=True)
class FrameMetadata:
    """Metadata for one extracted frame."""

    file_name: str
    output_path: str
    source_video: str
    video_stem: str
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class ExtractionResult:
    """Summary of a frame extraction run."""

    output_dir: Path
    metadata_path: Path
    video_count: int
    frame_count: int


@dataclass(frozen=True)
class SamplePoint:
    """A source frame location selected for extraction."""

    frame_index: int
    timestamp_seconds: float


def extract_video_frames(
    input_dir: Path,
    output_dir: Path,
    fps: float = 1.0,
    every_n_frames: int | None = None,
    image_format: str = "jpg",
    quality: int = 95,
    overwrite: bool = False,
    ffmpeg_path: str = "ffmpeg",
    ffprobe_path: str = "ffprobe",
) -> ExtractionResult:
    """Extract sampled frames from MP4 videos and write JSONL metadata."""
    normalized_format = image_format.lower().lstrip(".")
    _validate_options(fps, every_n_frames, normalized_format, quality)

    video_paths = list_video_paths(input_dir)
    if not video_paths:
        raise ValueError(f"No MP4 videos found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILE_NAME

    records: list[FrameMetadata] = []
    for video_path in video_paths:
        video_info = probe_video(video_path, ffprobe_path=ffprobe_path)
        for sample_point in iter_sample_points(video_info, fps, every_n_frames):
            frame_path = output_dir / build_frame_file_name(
                video_path.stem,
                sample_point.frame_index,
                sample_point.timestamp_seconds,
                normalized_format,
            )
            extract_frame(
                video_path=video_path,
                output_path=frame_path,
                timestamp_seconds=sample_point.timestamp_seconds,
                image_format=normalized_format,
                quality=quality,
                overwrite=overwrite,
                ffmpeg_path=ffmpeg_path,
            )
            records.append(
                FrameMetadata(
                    file_name=frame_path.name,
                    output_path=frame_path.relative_to(output_dir).as_posix(),
                    source_video=video_path.as_posix(),
                    video_stem=video_path.stem,
                    frame_index=sample_point.frame_index,
                    timestamp_seconds=round(sample_point.timestamp_seconds, 6),
                    width=video_info.width,
                    height=video_info.height,
                )
            )

    write_metadata(metadata_path, records)
    return ExtractionResult(
        output_dir=output_dir,
        metadata_path=metadata_path,
        video_count=len(video_paths),
        frame_count=len(records),
    )


def list_video_paths(input_dir: Path) -> list[Path]:
    """Return supported videos under an input directory in deterministic order."""
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def probe_video(video_path: Path, ffprobe_path: str = "ffprobe") -> VideoInfo:
    """Read video dimensions, frame rate, duration, and frame count with ffprobe."""
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")

    stream = streams[0]
    native_fps = _parse_frame_rate(
        stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    )
    duration_seconds = float(stream.get("duration") or 0)
    raw_frame_count = stream.get("nb_frames")
    frame_count = int(raw_frame_count) if raw_frame_count else None

    if native_fps <= 0:
        raise ValueError(f"Could not determine frame rate for {video_path}")
    if duration_seconds <= 0 and frame_count is None:
        raise ValueError(
            f"Could not determine duration or frame count for {video_path}"
        )

    return VideoInfo(
        path=video_path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=native_fps,
        duration_seconds=duration_seconds,
        frame_count=frame_count,
    )


def iter_sample_points(
    video_info: VideoInfo,
    fps: float,
    every_n_frames: int | None,
) -> Iterable[SamplePoint]:
    """Yield sample points for either time-based or frame-interval sampling."""
    if every_n_frames is not None:
        total_frames = video_info.frame_count
        if total_frames is None:
            total_frames = math.floor(video_info.duration_seconds * video_info.fps)
        for frame_index in range(0, total_frames, every_n_frames):
            yield SamplePoint(
                frame_index=frame_index,
                timestamp_seconds=frame_index / video_info.fps,
            )
        return

    interval_seconds = 1.0 / fps
    sample_count = max(1, math.ceil(video_info.duration_seconds / interval_seconds))
    for sample_number in range(sample_count):
        timestamp_seconds = sample_number * interval_seconds
        if timestamp_seconds >= video_info.duration_seconds:
            break
        frame_index = math.floor(timestamp_seconds * video_info.fps)
        if video_info.frame_count is not None and frame_index >= video_info.frame_count:
            break
        yield SamplePoint(
            frame_index=frame_index,
            timestamp_seconds=frame_index / video_info.fps,
        )


def build_frame_file_name(
    video_stem: str,
    frame_index: int,
    timestamp_seconds: float,
    image_format: str,
) -> str:
    """Build a deterministic filename for an extracted video frame."""
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_stem).strip("._")
    timestamp_ms = round(timestamp_seconds * 1000)
    return f"{safe_stem}_frame{frame_index:06d}_{timestamp_ms:010d}ms.{image_format}"


def extract_frame(
    video_path: Path,
    output_path: Path,
    timestamp_seconds: float,
    image_format: str,
    quality: int,
    overwrite: bool,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Extract one frame image at a timestamp using ffmpeg."""
    if output_path.exists() and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-ss",
        f"{timestamp_seconds:.6f}",
        "-frames:v",
        "1",
    ]
    if image_format in {"jpg", "jpeg"}:
        command.extend(["-q:v", str(_jpeg_quality_to_qscale(quality))])
    command.append(str(output_path))

    subprocess.run(command, check=True)


def write_metadata(metadata_path: Path, records: Iterable[FrameMetadata]) -> None:
    """Write extraction metadata as newline-delimited JSON."""
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for record in records:
            metadata_file.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def _validate_options(
    fps: float,
    every_n_frames: int | None,
    image_format: str,
    quality: int,
) -> None:
    """Validate frame extraction options before invoking ffmpeg."""
    if fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if every_n_frames is not None and every_n_frames <= 0:
        raise ValueError("--every-n-frames must be greater than 0")
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        raise ValueError(f"--image-format must be one of: {supported}")
    if not 1 <= quality <= 100:
        raise ValueError("--quality must be between 1 and 100")


def _parse_frame_rate(raw_frame_rate: str | None) -> float:
    """Parse an ffprobe frame-rate string into frames per second."""
    if not raw_frame_rate:
        return 0.0
    if "/" in raw_frame_rate:
        return float(Fraction(raw_frame_rate))
    return float(raw_frame_rate)


def _jpeg_quality_to_qscale(quality: int) -> int:
    """Convert JPEG quality to ffmpeg qscale value."""
    return max(2, min(31, round(31 - ((quality - 1) * 29 / 99))))
