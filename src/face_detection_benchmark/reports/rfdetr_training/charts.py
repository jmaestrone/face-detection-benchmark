"""SVG chart writers for RF-DETR training reports."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Any

from face_detection_benchmark.reports.rfdetr_training.parsing import has_metric_column
from face_detection_benchmark.reports.rfdetr_training.types import RfdetrTrainingRun


def _chart_color(series_index: int) -> str:
    """Return a stable color for one chart series."""
    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
    )
    return colors[series_index % len(colors)]


def _format_axis_value(value: float) -> str:
    """Format SVG axis labels compactly."""
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.1e}"
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _value_domain(
    values: list[float],
    lower_floor: float | None = None,
) -> tuple[float, float]:
    """Build a padded chart domain from numeric values."""
    minimum_value = min(values)
    maximum_value = max(values)
    if lower_floor is not None:
        minimum_value = min(minimum_value, lower_floor)
    if math.isclose(minimum_value, maximum_value):
        padding = 1.0 if math.isclose(maximum_value, 0.0) else abs(maximum_value) * 0.1
    else:
        padding = (maximum_value - minimum_value) * 0.08
    lower = minimum_value - padding
    upper = maximum_value + padding
    if lower_floor is not None:
        lower = max(lower_floor, lower)
    if math.isclose(lower, upper):
        upper = lower + 1.0
    return (lower, upper)


def _ticks(domain: tuple[float, float], count: int) -> list[float]:
    """Return evenly spaced tick values for a chart domain."""
    lower, upper = domain
    if count <= 1:
        return [lower]
    step = (upper - lower) / (count - 1)
    return [lower + (index * step) for index in range(count)]


def _series_for_rows(
    label: str,
    metrics_rows: list[dict[str, float | int]],
    column_name: str,
    color: str,
    stroke_dasharray: str | None = None,
) -> dict[str, Any]:
    """Build one SVG line series from cleaned metrics rows."""
    points = [
        (float(metrics_row["epoch"]), float(metrics_row[column_name]))
        for metrics_row in metrics_rows
        if column_name in metrics_row
    ]
    point_titles = [
        (
            f"epoch={int(metrics_row['epoch'])}, "
            f"step={int(metrics_row['step'])}, "
            f"{column_name}={float(metrics_row[column_name]):.4f}"
        )
        for metrics_row in metrics_rows
        if column_name in metrics_row
    ]
    return {
        "label": label,
        "points": points,
        "point_titles": point_titles,
        "color": color,
        "stroke_dasharray": stroke_dasharray,
    }


def _empty_svg(title: str) -> str:
    """Render an empty chart SVG with a clear no-data message."""
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="160" '
            'viewBox="0 0 820 160">',
            '<rect width="100%" height="100%" fill="#ffffff" />',
            f'<text x="410" y="72" text-anchor="middle" font-size="22" '
            f'font-weight="700" fill="#0f172a">{escape(title)}</text>',
            '<text x="410" y="104" text-anchor="middle" font-size="14" '
            'fill="#475569">No matching metrics available</text>',
            "</svg>",
        ]
    )


def _line_chart_svg(
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    x_domain: tuple[float, float],
    y_domain: tuple[float, float],
) -> str:
    """Render a complete SVG line chart."""
    width = 820
    margin_left = 78
    margin_top = 54
    margin_right = 36
    plot_width = width - margin_left - margin_right
    plot_height = 360
    plot_bottom = margin_top + plot_height
    legend_row_height = 24
    legend_start_y = plot_bottom + 82
    height = legend_start_y + (len(series) * legend_row_height) + 20

    def map_x(value: float) -> float:
        return (
            margin_left
            + ((value - x_domain[0]) / (x_domain[1] - x_domain[0])) * plot_width
        )

    def map_y(value: float) -> float:
        return (
            margin_top
            + (1 - ((value - y_domain[0]) / (y_domain[1] - y_domain[0]))) * plot_height
        )

    svg_elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{width / 2:.2f}" y="32" text-anchor="middle" '
        'font-size="22" font-weight="700" fill="#0f172a">'
        f"{escape(title)}</text>",
    ]
    for x_value in _ticks(x_domain, count=6):
        x_position = map_x(x_value)
        svg_elements.extend(
            [
                f'<line x1="{x_position:.2f}" y1="{margin_top}" '
                f'x2="{x_position:.2f}" y2="{plot_bottom}" stroke="#e5e7eb" />',
                f'<text x="{x_position:.2f}" y="{plot_bottom + 22}" '
                'text-anchor="middle" font-size="12" fill="#475569">'
                f"{_format_axis_value(x_value)}</text>",
            ]
        )
    for y_value in _ticks(y_domain, count=6):
        y_position = map_y(y_value)
        svg_elements.extend(
            [
                f'<line x1="{margin_left}" y1="{y_position:.2f}" '
                f'x2="{margin_left + plot_width}" y2="{y_position:.2f}" '
                'stroke="#e5e7eb" />',
                f'<text x="{margin_left - 14}" y="{y_position + 4:.2f}" '
                'text-anchor="end" font-size="12" fill="#475569">'
                f"{_format_axis_value(y_value)}</text>",
            ]
        )
    svg_elements.extend(
        [
            f'<line x1="{margin_left}" y1="{plot_bottom}" '
            f'x2="{margin_left + plot_width}" y2="{plot_bottom}" '
            'stroke="#334155" stroke-width="1.5" />',
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{plot_bottom}" '
            'stroke="#334155" stroke-width="1.5" />',
        ]
    )
    for series_index, chart_series in enumerate(series):
        point_pairs = " ".join(
            f"{map_x(x_value):.2f},{map_y(y_value):.2f}"
            for x_value, y_value in chart_series["points"]
        )
        dash_attribute = (
            f'stroke-dasharray="{chart_series["stroke_dasharray"]}" '
            if chart_series.get("stroke_dasharray")
            else ""
        )
        svg_elements.append(
            f'<polyline fill="none" stroke="{chart_series["color"]}" '
            f'stroke-width="3" {dash_attribute}points="{point_pairs}" />'
        )
        for point_index, (x_value, y_value) in enumerate(chart_series["points"]):
            svg_elements.append(
                f'<circle cx="{map_x(x_value):.2f}" cy="{map_y(y_value):.2f}" '
                f'r="4.5" fill="#ffffff" stroke="{chart_series["color"]}" '
                'stroke-width="2">'
                f"<title>{escape(chart_series['point_titles'][point_index])}</title>"
                "</circle>"
            )
        legend_y = legend_start_y + (series_index * legend_row_height)
        svg_elements.extend(
            [
                f'<line x1="{margin_left}" y1="{legend_y}" '
                f'x2="{margin_left + 30}" y2="{legend_y}" '
                f'stroke="{chart_series["color"]}" stroke-width="3" '
                f"{dash_attribute}/>",
                f'<text x="{margin_left + 40}" y="{legend_y + 4}" '
                f'font-size="13" fill="#334155">{escape(chart_series["label"])}</text>',
            ]
        )
    svg_elements.extend(
        [
            f'<text x="{width / 2:.2f}" y="{plot_bottom + 48}" '
            f'text-anchor="middle" font-size="14" fill="#334155">{x_label}</text>',
            f'<text x="22" y="{margin_top + plot_height / 2:.2f}" '
            'text-anchor="middle" font-size="14" fill="#334155" '
            f'transform="rotate(-90 22 {margin_top + plot_height / 2:.2f})">'
            f"{y_label}</text>",
            "</svg>",
        ]
    )
    return "\n".join(svg_elements)


def _write_line_chart(
    title: str,
    y_label: str,
    series: list[dict[str, Any]],
    output_path: Path,
    y_domain: tuple[float, float] | None = None,
) -> None:
    """Write a chart SVG to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not series:
        output_path.write_text(_empty_svg(title), encoding="utf-8")
        return

    all_points = [point for chart_series in series for point in chart_series["points"]]
    x_domain = _value_domain([point[0] for point in all_points], lower_floor=0.0)
    resolved_y_domain = y_domain or _value_domain([point[1] for point in all_points])
    output_path.write_text(
        _line_chart_svg(
            title=title,
            x_label="Epoch",
            y_label=y_label,
            series=series,
            x_domain=x_domain,
            y_domain=resolved_y_domain,
        ),
        encoding="utf-8",
    )


def write_metric_chart(
    title: str,
    y_label: str,
    metrics_rows: list[dict[str, float | int]],
    column_names: list[str],
    output_path: Path,
    y_domain: tuple[float, float] | None = None,
) -> None:
    """Write a single-run RF-DETR training metric chart."""
    series = [
        _series_for_rows(
            label=column_name,
            metrics_rows=metrics_rows,
            column_name=column_name,
            color=_chart_color(series_index),
        )
        for series_index, column_name in enumerate(column_names)
    ]
    _write_line_chart(
        title=title,
        y_label=y_label,
        series=[chart_series for chart_series in series if chart_series["points"]],
        output_path=output_path,
        y_domain=y_domain,
    )


def write_runs_overlay_chart(
    title: str,
    y_label: str,
    training_runs: list[RfdetrTrainingRun],
    column_names: tuple[str, ...],
    output_path: Path,
    y_domain: tuple[float, float] | None = None,
) -> None:
    """Write a comparison overlay chart for multiple RF-DETR training runs."""
    series = []
    for run_index, training_run in enumerate(training_runs):
        for column_index, column_name in enumerate(column_names):
            label = (
                training_run.display_label
                if len(column_names) == 1
                else f"{training_run.display_label} {column_name}"
            )
            chart_series = _series_for_rows(
                label=label,
                metrics_rows=training_run.metrics.rows,
                column_name=column_name,
                color=_chart_color(run_index),
                stroke_dasharray="6 4" if column_index else None,
            )
            if chart_series["points"]:
                series.append(chart_series)
    _write_line_chart(
        title=title,
        y_label=y_label,
        series=series,
        output_path=output_path,
        y_domain=y_domain,
    )


def metric_columns_present(
    metrics_rows: list[dict[str, float | int]],
    column_names: tuple[str, ...],
) -> list[str]:
    """Return requested columns that are present in at least one row."""
    return [
        column_name
        for column_name in column_names
        if has_metric_column(metrics_rows, column_name)
    ]
