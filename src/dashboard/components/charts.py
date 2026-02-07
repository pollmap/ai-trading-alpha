"""Shared chart helper functions for the ATLAS dashboard.

Provides reusable Plotly figure factories and layout utilities
used across all dashboard pages.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go
import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── KST Timezone Offset ────────────────────────────────────────────
_KST_OFFSET: timezone = timezone(timedelta(hours=9))

# ── Common Plotly Layout (dark theme) ──────────────────────────────
CHART_LAYOUT: dict[str, object] = {
    "template": "plotly_dark",
    "paper_bgcolor": "#0E1117",
    "plot_bgcolor": "#0E1117",
    "font": {
        "family": "Inter, sans-serif",
        "size": 13,
        "color": "#FAFAFA",
    },
    "margin": {"l": 60, "r": 30, "t": 50, "b": 50},
    "legend": {
        "bgcolor": "rgba(0,0,0,0)",
        "borderwidth": 0,
        "font": {"size": 11},
    },
    "hoverlabel": {
        "bgcolor": "#1E1E2E",
        "font_size": 12,
        "font_family": "Inter, sans-serif",
    },
    "xaxis": {
        "gridcolor": "#262730",
        "zerolinecolor": "#262730",
    },
    "yaxis": {
        "gridcolor": "#262730",
        "zerolinecolor": "#262730",
    },
}


def format_kst(utc_dt: datetime) -> str:
    """Convert a UTC datetime to a KST display string.

    Args:
        utc_dt: Timezone-aware UTC datetime.

    Returns:
        Formatted string in ``YYYY-MM-DD HH:MM KST``.
    """
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    kst_dt: datetime = utc_dt.astimezone(_KST_OFFSET)
    return kst_dt.strftime("%Y-%m-%d %H:%M KST")


def _apply_base_layout(fig: go.Figure, title: str) -> go.Figure:
    """Apply the shared CHART_LAYOUT to a figure and set its title.

    Args:
        fig: Plotly figure to update.
        title: Chart title text.

    Returns:
        The same figure (mutated in place) for chaining convenience.
    """
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    return fig


# ── Public Chart Factories ─────────────────────────────────────────


def create_equity_curve(
    data: dict[str, dict[str, list[datetime] | list[float]]],
    colors: dict[str, str],
    title: str,
) -> go.Figure:
    """Create an equity-curve line chart for multiple agents.

    ``data`` maps a display label to ``{"dates": [...], "values": [...]}``.
    Lines whose label contains ``"Multi"`` are rendered dashed; all others
    are solid.  ``colors`` maps the same labels to hex colour strings.

    Args:
        data: Mapping of agent label -> {"dates": list[datetime], "values": list[float]}.
        colors: Mapping of agent label -> hex colour.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig: go.Figure = go.Figure()

    for label, series in data.items():
        dates: list[datetime] = series["dates"]  # type: ignore[assignment]
        values: list[float] = series["values"]  # type: ignore[assignment]

        # Convert UTC dates to KST strings for display
        kst_dates: list[str] = [format_kst(dt) for dt in dates]

        dash_style: str = "dash" if "Multi" in label else "solid"
        color: str = colors.get(label, "#FFFFFF")

        fig.add_trace(
            go.Scatter(
                x=kst_dates,
                y=values,
                mode="lines",
                name=label,
                line={"color": color, "width": 2, "dash": dash_style},
                hovertemplate="%{x}<br>%{y:$,.0f}<extra>%{fullData.name}</extra>",
            )
        )

    _apply_base_layout(fig, title)
    fig.update_layout(
        yaxis_title="Portfolio Value",
        xaxis_title="Time (KST)",
        hovermode="x unified",
        height=500,
    )
    logger.debug("equity_curve_created", title=title, series_count=len(data))
    return fig


def create_heatmap(
    data: list[list[float]],
    x_labels: list[str],
    y_labels: list[str],
    title: str,
) -> go.Figure:
    """Create a heatmap chart.

    Args:
        data: 2-D list of numeric values (rows correspond to y_labels).
        x_labels: Column labels.
        y_labels: Row labels.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig: go.Figure = go.Figure(
        data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0.0, "#EF4444"],
                [0.5, "#1E1E2E"],
                [1.0, "#22C55E"],
            ],
            hovertemplate="Symbol: %{x}<br>Category: %{y}<br>Value: %{z:.2f}%<extra></extra>",
            texttemplate="%{z:.1f}%",
            textfont={"size": 11},
        )
    )

    _apply_base_layout(fig, title)
    fig.update_layout(height=400)
    logger.debug("heatmap_created", title=title)
    return fig


def create_radar_chart(
    categories: list[str],
    values_dict: dict[str, list[float]],
    title: str,
) -> go.Figure:
    """Create a radar (spider) chart comparing multiple series.

    Args:
        categories: Axis labels placed around the perimeter.
        values_dict: Mapping of series label -> list of values (same length as *categories*).
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig: go.Figure = go.Figure()

    for label, values in values_dict.items():
        # Close the polygon by repeating the first value
        closed_values: list[float] = values + [values[0]]
        closed_cats: list[str] = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=closed_values,
                theta=closed_cats,
                fill="toself",
                name=label,
                opacity=0.6,
            )
        )

    _apply_base_layout(fig, title)
    fig.update_layout(
        polar={
            "bgcolor": "#0E1117",
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "gridcolor": "#262730",
            },
            "angularaxis": {
                "gridcolor": "#262730",
            },
        },
        height=450,
    )
    logger.debug("radar_chart_created", title=title, series_count=len(values_dict))
    return fig
