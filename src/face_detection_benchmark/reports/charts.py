"""SVG chart writers for benchmark validation reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from face_detection_benchmark.evaluation.types import ThresholdValidationResult


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
    stroke_dasharray: str | None = None


def _line_chart_svg(
    title: str,
    x_label: str,
    y_label: str,
    series: list[ChartSeries],
    selected_points: list[tuple[tuple[float, float], str, str]],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
    tick_step: float,
) -> str:
    """Render a complete SVG line chart with grid, points, and legend."""
    width = 820
    height = 520
    margin_left = 78
    margin_top = 54
    margin_right = 36
    margin_bottom = 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def map_x(value: float) -> float:
        """Map a chart x value into SVG coordinates."""
        domain_lower, domain_upper = x_domain
        return (
            margin_left
            + ((value - domain_lower) / (domain_upper - domain_lower)) * plot_width
        )

    def map_y(value: float) -> float:
        """Map a chart y value into SVG coordinates."""
        domain_lower, domain_upper = y_domain
        return (
            margin_top
            + (1 - ((value - domain_lower) / (domain_upper - domain_lower)))
            * plot_height
        )

    grid_lines = []
    for x_value in _tick_values(x_domain, tick_step):
        x_position = map_x(x_value)
        grid_lines.extend(
            [
                f'<line x1="{x_position:.2f}" y1="{margin_top}" '
                f'x2="{x_position:.2f}" '
                f'y2="{margin_top + plot_height}" stroke="#e5e7eb" />',
                f'<text x="{x_position:.2f}" y="{height - 42}" '
                'text-anchor="middle" '
                f'font-size="12" fill="#475569">'
                f"{_format_tick(x_value, tick_step)}</text>",
            ]
        )
    for y_value in _tick_values(y_domain, tick_step):
        y_position = map_y(y_value)
        grid_lines.extend(
            [
                f'<line x1="{margin_left}" y1="{y_position:.2f}" '
                f'x2="{margin_left + plot_width}" y2="{y_position:.2f}" '
                'stroke="#e5e7eb" />',
                f'<text x="{margin_left - 14}" y="{y_position + 4:.2f}" '
                'text-anchor="end" font-size="12" fill="#475569">'
                f"{_format_tick(y_value, tick_step)}</text>",
            ]
        )

    polyline_elements = []
    point_elements = []
    point_label_elements = []
    legend_items = []
    for index, chart_series in enumerate(series):
        path_points = " ".join(
            f"{map_x(x_value):.2f},{map_y(y_value):.2f}"
            for x_value, y_value in chart_series.points
        )
        polyline_elements.append(
            f'<polyline fill="none" stroke="{chart_series.color}" stroke-width="3" '
            f'{_stroke_dash_attribute(chart_series)}points="{path_points}" />'
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
                f'stroke-width="3" {_stroke_dash_attribute(chart_series)}/>',
                f'<text x="{width - 146}" y="{legend_y + 4}" font-size="13" '
                f'fill="#334155">{escape(chart_series.label)}</text>',
            ]
        )

    selected_point_elements = []
    for selected_point, selected_label, selected_color in selected_points:
        selected_data_x, selected_data_y = selected_point
        selected_svg_x = map_x(selected_data_x)
        selected_svg_y = map_y(selected_data_y)
        selected_label_x = _clamp(
            selected_svg_x + 10,
            lower=margin_left + 4,
            upper=margin_left + plot_width - _label_width(selected_label) - 4,
        )
        selected_label_y = _clamp(
            selected_svg_y + 22,
            lower=margin_top + 16,
            upper=margin_top + plot_height - 8,
        )
        selected_point_elements.extend(
            [
                f'<circle cx="{selected_svg_x:.2f}" cy="{selected_svg_y:.2f}" '
                f'r="7" fill="{selected_color}" stroke="#334155" stroke-width="2">'
                f"<title>{escape(selected_label)}</title></circle>",
                f'<text x="{selected_label_x:.2f}" y="{selected_label_y:.2f}" '
                'font-size="13" font-weight="700" fill="#334155">'
                f"{escape(selected_label)}</text>",
            ]
        )
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
            *polyline_elements,
            *point_elements,
            *point_label_elements,
            *selected_point_elements,
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


def _precision_recall_series(
    result: ThresholdValidationResult,
    min_recall: float,
    min_precision: float,
    label: str = "precision_recall",
    color: str = "#2563eb",
    show_point_labels: bool = True,
    label_callouts: bool = True,
) -> ChartSeries:
    """Build one annotated series for the precision-recall chart."""
    precision_recall_groups = _group_threshold_rows_by_precision_recall(
        result.threshold_metrics
    )
    selected_point = (
        round(float(result.selected_metrics["recall"]), 6),
        round(float(result.selected_metrics["precision"]), 6),
    )
    visible_groups = [
        group
        for group in precision_recall_groups
        if (
            float(group["point"][0]) >= min_recall
            and float(group["point"][1]) >= min_precision
        )
        or group["point"] == selected_point
    ]
    if not visible_groups:
        visible_groups = precision_recall_groups
    return ChartSeries(
        label=label,
        points=[group["point"] for group in visible_groups],
        color=color,
        point_titles=[str(group["title"]) for group in visible_groups],
        point_labels=[str(group["label"]) for group in visible_groups],
        show_point_labels=show_point_labels,
        label_callouts=label_callouts,
    )


def _group_threshold_rows_by_precision_recall(
    threshold_metrics: list[dict[str, float | int]],
) -> list[dict[str, Any]]:
    """Collapse thresholds that land on the same precision-recall point."""
    grouped_rows: dict[tuple[float, float], list[dict[str, float | int]]] = {}
    for threshold_row in threshold_metrics:
        precision_recall_point = (
            round(float(threshold_row["recall"]), 6),
            round(float(threshold_row["precision"]), 6),
        )
        grouped_rows.setdefault(precision_recall_point, []).append(threshold_row)

    precision_recall_groups = []
    for precision_recall_point, rows in grouped_rows.items():
        first_row = rows[0]
        precision_recall_groups.append(
            {
                "point": precision_recall_point,
                "title": _threshold_group_title(rows),
                "label": _threshold_group_label(rows),
                "sort_threshold": float(first_row["confidence_threshold"]),
            }
        )
    return sorted(
        precision_recall_groups,
        key=lambda group: float(group["sort_threshold"]),
    )


def _threshold_score_points(
    result: ThresholdValidationResult,
    score_metric: str,
) -> list[tuple[float, float]]:
    """Build threshold-score points for an F-score series."""
    return [
        (
            float(threshold_row["confidence_threshold"]),
            float(threshold_row[score_metric]),
        )
        for threshold_row in result.threshold_metrics
    ]


def _threshold_score_title(row: dict[str, float | int], score_metric: str) -> str:
    """Format a tooltip title for one threshold score point."""
    return (
        f"threshold={_format_threshold(float(row['confidence_threshold']))}, "
        f"{score_metric}={float(row[score_metric]):.4f}"
    )


def _threshold_label(row: dict[str, float | int]) -> str:
    """Format a compact threshold point label."""
    return f"t={_format_threshold(float(row['confidence_threshold']))}"


def _threshold_group_title(rows: list[dict[str, float | int]]) -> str:
    """Format a tooltip title for grouped precision-recall thresholds."""
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
    """Format a label for one threshold or a same-point threshold range."""
    if len(rows) == 1:
        return _threshold_label(rows[0])
    first_threshold = float(rows[0]["confidence_threshold"])
    last_threshold = float(rows[-1]["confidence_threshold"])
    return f"t={_format_threshold(first_threshold)}-{_format_threshold(last_threshold)}"


def _format_threshold(threshold: float) -> str:
    """Format thresholds with enough precision for chart labels."""
    if 0.0 < abs(threshold) < 0.01:
        return f"{threshold:.3f}"
    return f"{threshold:.2f}"


def _point_label_offset(point_index: int) -> tuple[float, float]:
    """Return a repeating offset for visible point labels."""
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
    """Expand metric values to a rounded chart domain."""
    tick_step = 0.05
    domain_lower = max(0.0, math.floor(min(values) / tick_step) * tick_step)
    domain_upper = min(1.0, math.ceil(max(values) / tick_step) * tick_step)
    if domain_lower == domain_upper:
        domain_lower = max(0.0, domain_lower - tick_step)
        domain_upper = min(1.0, domain_upper + tick_step)
    return (round(domain_lower, 2), round(domain_upper, 2))


def _tick_values(domain: tuple[float, float], tick_step: float) -> list[float]:
    """Return rounded tick values for a chart domain."""
    domain_lower, domain_upper = domain
    start_index = math.ceil((domain_lower - 1e-9) / tick_step)
    end_index = math.floor((domain_upper + 1e-9) / tick_step)
    return [
        round(tick_index * tick_step, 2)
        for tick_index in range(start_index, end_index + 1)
    ]


def _format_tick(value: float, tick_step: float) -> str:
    """Format a chart tick for the configured spacing."""
    if tick_step < 0.1 and not math.isclose(value * 10, round(value * 10)):
        return f"{value:.2f}"
    return f"{value:.1f}"


def _clamp(value: float, lower: float, upper: float) -> float:
    """Constrain a numeric value to inclusive bounds."""
    return min(max(value, lower), upper)


def _label_width(label: str) -> float:
    """Estimate the rendered width of a point label."""
    return 8 + (len(label) * 6.1)


def _label_background_elements(
    label: str,
    label_x: float,
    label_y: float,
    label_width: float,
    point_x: float,
    point_y: float,
) -> list[str]:
    """Render callout line and background for a point label."""
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


def _model_color(index: int) -> str:
    """Return a stable chart color for a model series."""
    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
    )
    return colors[index % len(colors)]


def _stroke_dash_attribute(chart_series: ChartSeries) -> str:
    """Return an SVG dash attribute for dashed series."""
    if chart_series.stroke_dasharray is None:
        return ""
    return f'stroke-dasharray="{chart_series.stroke_dasharray}" '


def write_precision_recall_svg(
    result: ThresholdValidationResult,
    precision_recall_path: Path,
) -> None:
    """Write a precision-recall SVG chart for threshold validation results."""
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
            selected_points=[
                (
                    selected_point,
                    f"selected t={_format_threshold(result.selected_threshold)}",
                    "#f59e0b",
                )
            ],
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


def write_f_scores_svg(result: ThresholdValidationResult, f_scores_path: Path) -> None:
    """Write an F1/F2 threshold sweep SVG chart for validation results."""
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
                    stroke_dasharray="6 4",
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
            selected_points=[
                (
                    (
                        float(result.selected_metrics["confidence_threshold"]),
                        float(result.selected_metrics[selected_score_metric]),
                    ),
                    f"selected t={_format_threshold(result.selected_threshold)}",
                    "#f59e0b",
                )
            ],
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            tick_step=0.1,
        ),
        encoding="utf-8",
    )


def write_precision_recall_overlay_svg(
    results: list[ThresholdValidationResult],
    precision_recall_path: Path,
) -> None:
    """Write a multi-model precision-recall SVG chart."""
    series = [
        _precision_recall_series(
            result=result,
            min_recall=0.60,
            min_precision=0.60,
            label=result.model_name,
            color=_model_color(index),
            show_point_labels=False,
            label_callouts=False,
        )
        for index, result in enumerate(results)
    ]
    all_points = [point for chart_series in series for point in chart_series.points]
    selected_points = [
        (
            (
                float(result.selected_metrics["recall"]),
                float(result.selected_metrics["precision"]),
            ),
            f"{result.model_name} t={_format_threshold(result.selected_threshold)}",
            _model_color(index),
        )
        for index, result in enumerate(results)
    ]
    precision_recall_path.write_text(
        _line_chart_svg(
            title="Precision vs Recall Comparison",
            x_label="Recall",
            y_label="Precision",
            series=series,
            selected_points=selected_points,
            x_domain=_metric_domain([point[0] for point in all_points]),
            y_domain=_metric_domain([point[1] for point in all_points]),
            tick_step=0.05,
        ),
        encoding="utf-8",
    )


def write_f_scores_overlay_svg(
    results: list[ThresholdValidationResult],
    f_scores_path: Path,
) -> None:
    """Write a multi-model F1/F2 threshold sweep SVG chart."""
    series: list[ChartSeries] = []
    for index, result in enumerate(results):
        color = _model_color(index)
        series.append(
            ChartSeries(
                label=f"{result.model_name} f1",
                points=_threshold_score_points(result, "f1"),
                color=color,
                point_titles=[
                    _threshold_score_title(row, "f1")
                    for row in result.threshold_metrics
                ],
                point_labels=[
                    _threshold_label(row) for row in result.threshold_metrics
                ],
            )
        )
        series.append(
            ChartSeries(
                label=f"{result.model_name} f2",
                points=_threshold_score_points(result, "f2"),
                color=color,
                point_titles=[
                    _threshold_score_title(row, "f2")
                    for row in result.threshold_metrics
                ],
                point_labels=[
                    _threshold_label(row) for row in result.threshold_metrics
                ],
                stroke_dasharray="6 4",
            )
        )

    selected_points = [
        (
            (
                float(result.selected_metrics["confidence_threshold"]),
                float(
                    result.selected_metrics[
                        result.selection_metric
                        if result.selection_metric in {"f1", "f2"}
                        else "f2"
                    ]
                ),
            ),
            f"{result.model_name} t={_format_threshold(result.selected_threshold)}",
            _model_color(index),
        )
        for index, result in enumerate(results)
    ]
    f_scores_path.write_text(
        _line_chart_svg(
            title="F Scores by Threshold Comparison",
            x_label="Confidence threshold",
            y_label="Score",
            series=series,
            selected_points=selected_points,
            x_domain=(0.0, 1.0),
            y_domain=(0.0, 1.0),
            tick_step=0.1,
        ),
        encoding="utf-8",
    )
