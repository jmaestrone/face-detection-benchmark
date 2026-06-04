"""Report writers for benchmark evaluation outputs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from face_detection_benchmark.evaluation import (
    DetectionMetrics,
    ThresholdValidationResult,
    metrics_to_json_dict,
    threshold_validation_to_json_dict,
)


@dataclass(frozen=True)
class ChartSeries:
    """One line and its annotated points in a generated SVG chart."""

    label: str
    points: list[tuple[float, float]]
    color: str
    point_titles: list[str]
    point_labels: list[str]
    show_point_labels: bool = False
    label_callouts: bool = False


def write_evaluation_reports(
    metrics: DetectionMetrics,
    output_dir: Path,
    results_table_path: Path | None = None,
    leaderboard_path: Path | None = None,
    run_id: str | None = None,
    dataset_dir: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Path]:
    """Write per-run reports and optionally append a cumulative results row."""
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{metrics.model_name}.json"
    summary_path = output_dir / "summary.csv"
    sweep_path = metrics_dir / f"{metrics.model_name}_confidence_sweep.csv"

    metrics_path.write_text(
        json.dumps(metrics_to_json_dict(metrics), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_csv(metrics, summary_path)

    report_paths = {
        "metrics_path": metrics_path,
        "summary_path": summary_path,
    }
    if metrics.confidence_sweep:
        _write_sweep_csv(metrics, sweep_path)
        report_paths["sweep_path"] = sweep_path
    if results_table_path is not None:
        append_results_row(
            metrics=metrics,
            results_table_path=results_table_path,
            run_id=run_id,
            dataset_dir=dataset_dir,
            predictions_path=predictions_path,
        )
        report_paths["results_table_path"] = results_table_path
        if leaderboard_path is not None:
            write_results_leaderboard(
                results_table_path=results_table_path,
                leaderboard_path=leaderboard_path,
            )
            report_paths["leaderboard_path"] = leaderboard_path
    return report_paths


def write_threshold_validation_reports(
    result: ThresholdValidationResult,
    output_dir: Path,
    dataset_dir: Path | None = None,
    predictions_path: Path | None = None,
) -> dict[str, Path]:
    """Write validation threshold tables, selection metadata, and SVG plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "threshold_validation.json"
    selected_threshold_path = output_dir / "selected_threshold.json"
    threshold_metrics_path = output_dir / "threshold_metrics.csv"
    threshold_metrics_markdown_path = output_dir / "threshold_metrics.md"
    precision_recall_path = output_dir / "precision_recall.svg"
    f_scores_path = output_dir / "f1_f2_by_threshold.svg"

    validation_payload = threshold_validation_to_json_dict(result)
    validation_payload["dataset_dir"] = dataset_dir.as_posix() if dataset_dir else ""
    validation_payload["predictions_path"] = (
        predictions_path.as_posix() if predictions_path else ""
    )
    validation_path.write_text(
        json.dumps(validation_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    selected_threshold_path.write_text(
        json.dumps(
            {
                "model_name": result.model_name,
                "selection_metric": result.selection_metric,
                "selected_threshold": result.selected_threshold,
                "selected_metrics": result.selected_metrics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_threshold_metrics_csv(result, threshold_metrics_path)
    _write_threshold_metrics_markdown(result, threshold_metrics_markdown_path)
    _write_precision_recall_svg(result, precision_recall_path)
    _write_f_scores_svg(result, f_scores_path)
    return {
        "validation_path": validation_path,
        "selected_threshold_path": selected_threshold_path,
        "threshold_metrics_path": threshold_metrics_path,
        "threshold_metrics_markdown_path": threshold_metrics_markdown_path,
        "precision_recall_path": precision_recall_path,
        "f_scores_path": f_scores_path,
    }


def append_results_row(
    metrics: DetectionMetrics,
    results_table_path: Path,
    run_id: str | None,
    dataset_dir: Path | None,
    predictions_path: Path | None,
) -> None:
    """Append one evaluation row to the cumulative local results table."""
    results_table_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "model_name",
        "dataset_dir",
        "predictions_path",
        "image_count",
        "ground_truth_count",
        "prediction_count",
        "confidence_threshold",
        "iou_threshold",
        "true_positive_count",
        "false_positive_count",
        "false_negative_count",
        "precision",
        "recall",
        "f1",
        "f2",
        "ap50",
        "ap75",
        "map_50_95",
    ]
    should_write_header = not results_table_path.exists()
    with results_table_path.open("a", encoding="utf-8", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id or "",
                "model_name": metrics.model_name,
                "dataset_dir": dataset_dir.as_posix() if dataset_dir else "",
                "predictions_path": (
                    predictions_path.as_posix() if predictions_path else ""
                ),
                "image_count": metrics.image_count,
                "ground_truth_count": metrics.ground_truth_count,
                "prediction_count": metrics.prediction_count,
                "confidence_threshold": metrics.confidence_threshold,
                "iou_threshold": metrics.iou_threshold,
                "true_positive_count": metrics.true_positive_count,
                "false_positive_count": metrics.false_positive_count,
                "false_negative_count": metrics.false_negative_count,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "f2": metrics.f2,
                "ap50": metrics.ap50,
                "ap75": metrics.ap75,
                "map_50_95": metrics.map_50_95,
            }
        )


def write_results_leaderboard(
    results_table_path: Path,
    leaderboard_path: Path,
) -> None:
    """Write a human-readable Markdown leaderboard from the results CSV."""
    rows = _read_results_rows(results_table_path)
    rows = sorted(
        rows,
        key=lambda row: (
            _float_value(row, "map_50_95"),
            _float_value(row, "f2"),
            _float_value(row, "ap50"),
        ),
        reverse=True,
    )
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(
        "\n".join(_leaderboard_lines(rows)) + "\n",
        encoding="utf-8",
    )


def _read_results_rows(results_table_path: Path) -> list[dict[str, Any]]:
    if not results_table_path.exists():
        return []
    with results_table_path.open("r", encoding="utf-8", newline="") as results_file:
        return list(csv.DictReader(results_file))


def _leaderboard_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# Face Detection Benchmark Results",
        "",
        "Sorted by `mAP@[0.50:0.95]`, then F2, then AP50.",
        "",
        "| Rank | Run | Model | Threshold | Precision | Recall | F1 | F2 | AP50 | AP75 | mAP50-95 | TP | FP | FN |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row.get("run_id", "")),
                    str(row.get("model_name", "")),
                    _format_float(row.get("confidence_threshold", "")),
                    _format_float(row.get("precision", "")),
                    _format_float(row.get("recall", "")),
                    _format_float(row.get("f1", "")),
                    _format_float(row.get("f2", "")),
                    _format_float(row.get("ap50", "")),
                    _format_float(row.get("ap75", "")),
                    _format_float(row.get("map_50_95", "")),
                    str(row.get("true_positive_count", "")),
                    str(row.get("false_positive_count", "")),
                    str(row.get("false_negative_count", "")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "This file is generated from `runs/benchmarks/results.csv` and is ignored by git.",
        ]
    )
    return lines


def _float_value(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return ""


def _write_summary_csv(metrics: DetectionMetrics, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "model_name",
                "image_count",
                "ground_truth_count",
                "prediction_count",
                "confidence_threshold",
                "iou_threshold",
                "true_positive_count",
                "false_positive_count",
                "false_negative_count",
                "precision",
                "recall",
                "f1",
                "f2",
                "ap50",
                "ap75",
                "map_50_95",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model_name": metrics.model_name,
                "image_count": metrics.image_count,
                "ground_truth_count": metrics.ground_truth_count,
                "prediction_count": metrics.prediction_count,
                "confidence_threshold": metrics.confidence_threshold,
                "iou_threshold": metrics.iou_threshold,
                "true_positive_count": metrics.true_positive_count,
                "false_positive_count": metrics.false_positive_count,
                "false_negative_count": metrics.false_negative_count,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "f2": metrics.f2,
                "ap50": metrics.ap50,
                "ap75": metrics.ap75,
                "map_50_95": metrics.map_50_95,
            }
        )


def _write_sweep_csv(metrics: DetectionMetrics, sweep_path: Path) -> None:
    with sweep_path.open("w", encoding="utf-8", newline="") as sweep_file:
        fieldnames = [
            "confidence_threshold",
            "true_positive_count",
            "false_positive_count",
            "false_negative_count",
            "precision",
            "recall",
            "f1",
            "f2",
        ]
        writer = csv.DictWriter(sweep_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics.confidence_sweep)


def _write_threshold_metrics_csv(
    result: ThresholdValidationResult,
    threshold_metrics_path: Path,
) -> None:
    threshold_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with threshold_metrics_path.open(
        "w", encoding="utf-8", newline=""
    ) as threshold_metrics_file:
        writer = csv.DictWriter(
            threshold_metrics_file,
            fieldnames=[
                "confidence_threshold",
                "true_positive_count",
                "false_positive_count",
                "false_negative_count",
                "precision",
                "recall",
                "f1",
                "f2",
            ],
        )
        writer.writeheader()
        writer.writerows(result.threshold_metrics)


def _write_threshold_metrics_markdown(
    result: ThresholdValidationResult,
    threshold_metrics_markdown_path: Path,
) -> None:
    lines = [
        "# Validation Threshold Metrics",
        "",
        f"Model: `{result.model_name}`",
        f"Selection metric: `{result.selection_metric}`",
        f"Selected threshold: `{result.selected_threshold:.4f}`",
        f"IoU threshold: `{result.iou_threshold:.4f}`",
        "",
        "| Threshold | Precision | Recall | F1 | F2 | TP | FP | FN |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result.threshold_metrics:
        prefix = "**" if row == result.selected_metrics else ""
        suffix = "**" if row == result.selected_metrics else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{prefix}{float(row['confidence_threshold']):.4f}{suffix}",
                    f"{prefix}{float(row['precision']):.4f}{suffix}",
                    f"{prefix}{float(row['recall']):.4f}{suffix}",
                    f"{prefix}{float(row['f1']):.4f}{suffix}",
                    f"{prefix}{float(row['f2']):.4f}{suffix}",
                    f"{prefix}{int(row['true_positive_count'])}{suffix}",
                    f"{prefix}{int(row['false_positive_count'])}{suffix}",
                    f"{prefix}{int(row['false_negative_count'])}{suffix}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "This validation output can be used to choose an operating threshold. "
            "Do not also report the same data as an unbiased test benchmark.",
        ]
    )
    threshold_metrics_markdown_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_precision_recall_svg(
    result: ThresholdValidationResult,
    precision_recall_path: Path,
) -> None:
    precision_recall_series = _precision_recall_series(
        result=result,
        min_recall=0.60,
        min_precision=0.60,
    )
    selected_point = (
        float(result.selected_metrics["recall"]),
        float(result.selected_metrics["precision"]),
    )
    precision_recall_path.write_text(
        _line_chart_svg(
            title="Precision vs Recall",
            x_label="Recall",
            y_label="Precision",
            series=[precision_recall_series],
            selected_point=selected_point,
            selected_label=f"selected t={_format_threshold(result.selected_threshold)}",
            x_domain=_metric_domain(
                [point[0] for point in precision_recall_series.points]
            ),
            y_domain=_metric_domain(
                [point[1] for point in precision_recall_series.points]
            ),
            tick_step=0.05,
        ),
        encoding="utf-8",
    )


def _write_f_scores_svg(result: ThresholdValidationResult, f_scores_path: Path) -> None:
    selected_score_metric = (
        result.selection_metric if result.selection_metric in {"f1", "f2"} else "f2"
    )
    f_scores_path.write_text(
        _line_chart_svg(
            title="F Scores by Threshold",
            x_label="Confidence threshold",
            y_label="Score",
            series=[
                ChartSeries(
                    label="f1",
                    points=_threshold_score_points(result, "f1"),
                    color="#16a34a",
                    point_titles=[
                        _threshold_score_title(row, "f1")
                        for row in result.threshold_metrics
                    ],
                    point_labels=[
                        _threshold_label(row) for row in result.threshold_metrics
                    ],
                ),
                ChartSeries(
                    label="f2",
                    points=_threshold_score_points(result, "f2"),
                    color="#dc2626",
                    point_titles=[
                        _threshold_score_title(row, "f2")
                        for row in result.threshold_metrics
                    ],
                    point_labels=[
                        _threshold_label(row) for row in result.threshold_metrics
                    ],
                ),
            ],
            selected_point=(
                float(result.selected_metrics["confidence_threshold"]),
                float(result.selected_metrics[selected_score_metric]),
            ),
            selected_label=f"selected t={_format_threshold(result.selected_threshold)}",
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            tick_step=0.1,
        ),
        encoding="utf-8",
    )


def _line_chart_svg(
    title: str,
    x_label: str,
    y_label: str,
    series: list[ChartSeries],
    selected_point: tuple[float, float],
    selected_label: str,
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    tick_step: float,
) -> str:
    width = 820
    height = 520
    margin_left = 78
    margin_top = 54
    margin_right = 36
    margin_bottom = 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def map_x(value: float) -> float:
        lower, upper = x_domain
        return margin_left + ((value - lower) / (upper - lower)) * plot_width

    def map_y(value: float) -> float:
        lower, upper = y_domain
        return margin_top + (1 - ((value - lower) / (upper - lower))) * plot_height

    grid_lines = []
    for x_value in _tick_values(x_domain, tick_step):
        x = map_x(x_value)
        grid_lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
                f'y2="{margin_top + plot_height}" stroke="#e5e7eb" />',
                f'<text x="{x:.2f}" y="{height - 42}" text-anchor="middle" '
                f'font-size="12" fill="#475569">'
                f"{_format_tick(x_value, tick_step)}</text>",
            ]
        )
    for y_value in _tick_values(y_domain, tick_step):
        y = map_y(y_value)
        grid_lines.extend(
            [
                f'<line x1="{margin_left}" y1="{y:.2f}" '
                f'x2="{margin_left + plot_width}" y2="{y:.2f}" '
                'stroke="#e5e7eb" />',
                f'<text x="{margin_left - 14}" y="{y + 4:.2f}" '
                'text-anchor="end" font-size="12" fill="#475569">'
                f"{_format_tick(y_value, tick_step)}</text>",
            ]
        )

    paths = []
    point_elements = []
    point_label_elements = []
    legend_items = []
    for index, chart_series in enumerate(series):
        path_points = " ".join(
            f"{map_x(x_value):.2f},{map_y(y_value):.2f}"
            for x_value, y_value in chart_series.points
        )
        paths.append(
            f'<polyline fill="none" stroke="{chart_series.color}" stroke-width="3" '
            f'points="{path_points}" />'
        )
        for point_index, (x_value, y_value) in enumerate(chart_series.points):
            point_x = map_x(x_value)
            point_y = map_y(y_value)
            point_title = chart_series.point_titles[point_index]
            point_label = chart_series.point_labels[point_index]
            point_elements.append(
                f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="4.5" '
                f'fill="#ffffff" stroke="{chart_series.color}" stroke-width="2">'
                f"<title>{escape(point_title)}</title></circle>"
            )
            if chart_series.show_point_labels:
                label_offset_x, label_offset_y = _point_label_offset(point_index)
                label_width = _label_width(point_label)
                label_x = _clamp(
                    point_x + label_offset_x,
                    lower=margin_left + 4,
                    upper=margin_left + plot_width - label_width - 4,
                )
                label_y = _clamp(
                    point_y + label_offset_y,
                    lower=margin_top + 16,
                    upper=margin_top + plot_height - 8,
                )
                if chart_series.label_callouts:
                    label_background_elements = _label_background_elements(
                        label=point_label,
                        label_x=label_x,
                        label_y=label_y,
                        label_width=label_width,
                        point_x=point_x,
                        point_y=point_y,
                    )
                    point_label_elements.extend(label_background_elements)
                point_label_elements.append(
                    f'<text x="{label_x:.2f}" y="{label_y:.2f}" '
                    'font-size="11" fill="#1f2937">'
                    f"{escape(point_label)}</text>"
                )
        legend_y = margin_top + 6 + (index * 24)
        legend_items.extend(
            [
                f'<line x1="{width - 184}" y1="{legend_y}" '
                f'x2="{width - 154}" y2="{legend_y}" stroke="{chart_series.color}" '
                'stroke-width="3" />',
                f'<text x="{width - 146}" y="{legend_y + 4}" font-size="13" '
                f'fill="#334155">{escape(chart_series.label)}</text>',
            ]
        )

    selected_x, selected_y = selected_point
    selected_svg_x = map_x(selected_x)
    selected_svg_y = map_y(selected_y)
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="{width / 2:.2f}" y="32" text-anchor="middle" '
            'font-size="22" font-weight="700" '
            f'fill="#0f172a">{escape(title)}</text>',
            *grid_lines,
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
            f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" '
            'stroke="#334155" stroke-width="1.5" />',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
            f'y2="{margin_top + plot_height}" stroke="#334155" '
            'stroke-width="1.5" />',
            *paths,
            *point_elements,
            *point_label_elements,
            f'<circle cx="{selected_svg_x:.2f}" cy="{selected_svg_y:.2f}" '
            'r="7" fill="#f59e0b" stroke="#92400e" stroke-width="2">'
            f"<title>{escape(selected_label)}</title></circle>",
            f'<text x="{selected_svg_x + 10:.2f}" y="{selected_svg_y + 22:.2f}" '
            'font-size="13" font-weight="700" fill="#92400e">'
            f"{escape(selected_label)}</text>",
            f'<text x="{width / 2:.2f}" y="{height - 12}" '
            f'text-anchor="middle" font-size="14" fill="#334155">{x_label}</text>',
            f'<text x="22" y="{margin_top + plot_height / 2:.2f}" '
            'text-anchor="middle" font-size="14" fill="#334155" '
            f'transform="rotate(-90 22 {margin_top + plot_height / 2:.2f})">'
            f"{y_label}</text>",
            *legend_items,
            "</svg>",
        ]
    )


def _precision_recall_points(
    result: ThresholdValidationResult,
) -> list[tuple[float, float]]:
    return [
        (float(row["recall"]), float(row["precision"]))
        for row in result.threshold_metrics
    ]


def _precision_recall_series(
    result: ThresholdValidationResult,
    min_recall: float,
    min_precision: float,
) -> ChartSeries:
    groups = _group_threshold_rows_by_precision_recall(result.threshold_metrics)
    selected_point = (
        round(float(result.selected_metrics["recall"]), 6),
        round(float(result.selected_metrics["precision"]), 6),
    )
    visible_groups = [
        group
        for group in groups
        if (
            float(group["point"][0]) >= min_recall
            and float(group["point"][1]) >= min_precision
        )
        or group["point"] == selected_point
    ]
    if not visible_groups:
        visible_groups = groups
    return ChartSeries(
        label="precision_recall",
        points=[group["point"] for group in visible_groups],
        color="#2563eb",
        point_titles=[str(group["title"]) for group in visible_groups],
        point_labels=[str(group["label"]) for group in visible_groups],
        show_point_labels=True,
        label_callouts=True,
    )


def _group_threshold_rows_by_precision_recall(
    threshold_metrics: list[dict[str, float | int]],
) -> list[dict[str, Any]]:
    grouped_rows: dict[tuple[float, float], list[dict[str, float | int]]] = {}
    for row in threshold_metrics:
        point = (round(float(row["recall"]), 6), round(float(row["precision"]), 6))
        grouped_rows.setdefault(point, []).append(row)

    groups = []
    for point, rows in grouped_rows.items():
        first_row = rows[0]
        groups.append(
            {
                "point": point,
                "title": _threshold_group_title(rows),
                "label": _threshold_group_label(rows),
                "sort_threshold": float(first_row["confidence_threshold"]),
            }
        )
    return sorted(groups, key=lambda group: float(group["sort_threshold"]))


def _threshold_score_points(
    result: ThresholdValidationResult,
    score_metric: str,
) -> list[tuple[float, float]]:
    return [
        (float(row["confidence_threshold"]), float(row[score_metric]))
        for row in result.threshold_metrics
    ]


def _threshold_point_title(row: dict[str, float | int]) -> str:
    return (
        f"threshold={_format_threshold(float(row['confidence_threshold']))}, "
        f"precision={float(row['precision']):.4f}, "
        f"recall={float(row['recall']):.4f}, "
        f"f1={float(row['f1']):.4f}, "
        f"f2={float(row['f2']):.4f}"
    )


def _threshold_score_title(row: dict[str, float | int], score_metric: str) -> str:
    return (
        f"threshold={_format_threshold(float(row['confidence_threshold']))}, "
        f"{score_metric}={float(row[score_metric]):.4f}"
    )


def _threshold_label(row: dict[str, float | int]) -> str:
    return f"t={_format_threshold(float(row['confidence_threshold']))}"


def _threshold_group_title(rows: list[dict[str, float | int]]) -> str:
    first_row = rows[0]
    threshold_label = _threshold_group_label(rows).replace("t=", "threshold=")
    return (
        f"{threshold_label}, "
        f"precision={float(first_row['precision']):.4f}, "
        f"recall={float(first_row['recall']):.4f}, "
        f"f1={float(first_row['f1']):.4f}, "
        f"f2={float(first_row['f2']):.4f}"
    )


def _threshold_group_label(rows: list[dict[str, float | int]]) -> str:
    if len(rows) == 1:
        return _threshold_label(rows[0])
    first_threshold = float(rows[0]["confidence_threshold"])
    last_threshold = float(rows[-1]["confidence_threshold"])
    return f"t={_format_threshold(first_threshold)}-{_format_threshold(last_threshold)}"


def _format_threshold(threshold: float) -> str:
    if 0.0 < abs(threshold) < 0.01:
        return f"{threshold:.3f}"
    return f"{threshold:.2f}"


def _point_label_offset(point_index: int) -> tuple[float, float]:
    offsets = [
        (12.0, -14.0),
        (12.0, 24.0),
        (-96.0, -14.0),
        (-96.0, 24.0),
        (18.0, -36.0),
        (-104.0, 42.0),
    ]
    return offsets[point_index % len(offsets)]


def _metric_domain(values: list[float]) -> tuple[float, float]:
    tick_step = 0.05
    lower = max(0.0, math.floor(min(values) / tick_step) * tick_step)
    upper = min(1.0, math.ceil(max(values) / tick_step) * tick_step)
    if lower == upper:
        lower = max(0.0, lower - tick_step)
        upper = min(1.0, upper + tick_step)
    return (round(lower, 2), round(upper, 2))


def _tick_values(domain: tuple[float, float], tick_step: float) -> list[float]:
    lower, upper = domain
    start = math.ceil((lower - 1e-9) / tick_step)
    end = math.floor((upper + 1e-9) / tick_step)
    return [round(index * tick_step, 2) for index in range(start, end + 1)]


def _format_tick(value: float, tick_step: float) -> str:
    if tick_step < 0.1 and not math.isclose(value * 10, round(value * 10)):
        return f"{value:.2f}"
    return f"{value:.1f}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _label_width(label: str) -> float:
    return 8 + (len(label) * 6.1)


def _label_background_elements(
    label: str,
    label_x: float,
    label_y: float,
    label_width: float,
    point_x: float,
    point_y: float,
) -> list[str]:
    label_height = 17
    rect_x = label_x - 4
    rect_y = label_y - 13
    line_target_x = rect_x if label_x > point_x else rect_x + label_width
    line_target_y = rect_y + (label_height / 2)
    return [
        f'<line x1="{point_x:.2f}" y1="{point_y:.2f}" '
        f'x2="{line_target_x:.2f}" y2="{line_target_y:.2f}" '
        'stroke="#94a3b8" stroke-width="1" />',
        f'<rect x="{rect_x:.2f}" y="{rect_y:.2f}" '
        f'width="{label_width:.2f}" height="{label_height}" rx="3" '
        'fill="#ffffff" stroke="#cbd5e1" stroke-width="1" opacity="0.94" />',
    ]
