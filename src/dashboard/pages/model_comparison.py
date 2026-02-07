"""Model Comparison page — heatmap, radar chart, statistical significance table.

Compares 4 LLMs x 2 architectures on multiple performance dimensions.
Uses demo data when the database is not connected.
"""

from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from src.core.constants import (
    ARCH_MULTI,
    ARCH_SINGLE,
    MODEL_CLAUDE,
    MODEL_COLORS,
    MODEL_DEEPSEEK,
    MODEL_GEMINI,
    MODEL_GPT,
)

log: structlog.stdlib.BoundLogger = structlog.get_logger("dashboard.model_comparison")

# ── Timezone ────────────────────────────────────────────────────
KST: timezone = timezone(timedelta(hours=9))

# ── Constants ───────────────────────────────────────────────────
MODELS: list[str] = [MODEL_DEEPSEEK, MODEL_GEMINI, MODEL_CLAUDE, MODEL_GPT]
ARCHITECTURES: list[str] = [ARCH_SINGLE, ARCH_MULTI]

DISPLAY_NAMES: dict[str, str] = {
    MODEL_DEEPSEEK: "DeepSeek",
    MODEL_GEMINI: "Gemini",
    MODEL_CLAUDE: "Claude",
    MODEL_GPT: "GPT",
}

RADAR_METRICS: list[str] = [
    "Return (%)",
    "Sharpe Ratio",
    "MDD (abs, %)",
    "Win Rate (%)",
    "Cost Efficiency",
]


# ── Demo data generation ────────────────────────────────────────


@st.cache_data(ttl=60)
def _generate_demo_matrix_data() -> pd.DataFrame:
    """Generate a 4x2 matrix of model x architecture performance metrics.

    Returns:
        DataFrame with columns: model, architecture, total_return_pct,
        sharpe_ratio, max_drawdown_pct, win_rate_pct, cost_efficiency,
        total_trades, avg_confidence.
    """
    rng: np.random.Generator = np.random.default_rng(seed=123)
    rows: list[dict[str, object]] = []

    for model in MODELS:
        for arch in ARCHITECTURES:
            # Multi-agent architectures tend to have slightly better metrics
            arch_bonus: float = 1.5 if arch == ARCH_MULTI else 0.0
            total_ret: float = rng.uniform(-3.0, 12.0) + arch_bonus
            sharpe: float = rng.uniform(-0.3, 2.2) + (arch_bonus * 0.1)
            mdd: float = rng.uniform(-18.0, -3.0)
            win_rate: float = rng.uniform(42.0, 62.0) + (arch_bonus * 0.5)
            cost_eff: float = rng.uniform(0.3, 0.95)
            total_trades: int = int(rng.integers(30, 120))
            avg_conf: float = rng.uniform(0.45, 0.85)

            rows.append(
                {
                    "model": model,
                    "architecture": arch,
                    "total_return_pct": round(total_ret, 2),
                    "sharpe_ratio": round(sharpe, 2),
                    "max_drawdown_pct": round(mdd, 2),
                    "win_rate_pct": round(win_rate, 1),
                    "cost_efficiency": round(cost_eff, 3),
                    "total_trades": total_trades,
                    "avg_confidence": round(avg_conf, 3),
                }
            )

    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def _generate_demo_significance() -> pd.DataFrame:
    """Generate pairwise statistical significance (p-values) between models.

    Returns:
        DataFrame with columns: model_a, model_b, metric, p_value, significant.
    """
    rng: np.random.Generator = np.random.default_rng(seed=456)
    rows: list[dict[str, object]] = []

    for i, model_a in enumerate(MODELS):
        for model_b in MODELS[i + 1 :]:
            for metric in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct"]:
                p_val: float = rng.uniform(0.001, 0.5)
                rows.append(
                    {
                        "model_a": DISPLAY_NAMES[model_a],
                        "model_b": DISPLAY_NAMES[model_b],
                        "metric": metric,
                        "p_value": round(p_val, 4),
                        "significant": p_val < 0.05,
                    }
                )

    return pd.DataFrame(rows)


# ── Chart builders ──────────────────────────────────────────────


def _build_heatmap(matrix_df: pd.DataFrame) -> go.Figure:
    """Build a 4x2 heatmap of models (rows) vs architectures (columns).

    Colored by total return percentage.
    """
    # Pivot to matrix form
    pivot: pd.DataFrame = matrix_df.pivot(
        index="model", columns="architecture", values="total_return_pct"
    )
    # Reorder
    pivot = pivot.reindex(index=MODELS, columns=ARCHITECTURES)

    display_row_labels: list[str] = [DISPLAY_NAMES[m] for m in MODELS]

    # Build annotation text (return + sharpe)
    annotations: list[list[str]] = []
    for model in MODELS:
        row_annot: list[str] = []
        for arch in ARCHITECTURES:
            subset: pd.DataFrame = matrix_df[
                (matrix_df["model"] == model) & (matrix_df["architecture"] == arch)
            ]
            if not subset.empty:
                ret: float = subset["total_return_pct"].iloc[0]
                sharpe: float = subset["sharpe_ratio"].iloc[0]
                row_annot.append(f"{ret:+.2f}%<br>Sharpe: {sharpe:.2f}")
            else:
                row_annot.append("N/A")
        annotations.append(row_annot)

    fig: go.Figure = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[a.capitalize() for a in ARCHITECTURES],
            y=display_row_labels,
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 13},
            colorscale=[
                [0.0, "#FFCDD2"],  # negative returns: red-ish
                [0.5, "#FFFDE7"],  # around zero: light yellow
                [1.0, "#C8E6C9"],  # positive returns: green-ish
            ],
            colorbar=dict(title="Return (%)"),
            hovertemplate=(
                "Model: %{y}<br>"
                "Architecture: %{x}<br>"
                "Return: %{z:.2f}%<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Total Return by Model x Architecture",
        xaxis_title="Architecture",
        yaxis_title="Model",
        height=380,
        template="plotly_white",
        margin=dict(l=100, t=60),
    )

    return fig


def _build_radar_chart(matrix_df: pd.DataFrame) -> go.Figure:
    """Build a radar chart comparing all 4 models on 5 metrics.

    Aggregates across architectures (average of single + multi).
    """
    fig: go.Figure = go.Figure()

    for model in MODELS:
        subset: pd.DataFrame = matrix_df[matrix_df["model"] == model]
        if subset.empty:
            continue

        # Average across architectures
        avg_return: float = float(subset["total_return_pct"].mean())
        avg_sharpe: float = float(subset["sharpe_ratio"].mean())
        avg_mdd_abs: float = float(subset["max_drawdown_pct"].abs().mean())
        avg_win_rate: float = float(subset["win_rate_pct"].mean())
        avg_cost_eff: float = float(subset["cost_efficiency"].mean())

        # Normalize to 0-100 scale for radar chart display
        # Return: map [-10, 20] -> [0, 100]
        norm_return: float = max(0.0, min(100.0, (avg_return + 10.0) / 30.0 * 100.0))
        # Sharpe: map [-1, 3] -> [0, 100]
        norm_sharpe: float = max(0.0, min(100.0, (avg_sharpe + 1.0) / 4.0 * 100.0))
        # MDD (abs): lower is better, invert. map [0, 20] -> [100, 0]
        norm_mdd: float = max(0.0, min(100.0, (1.0 - avg_mdd_abs / 20.0) * 100.0))
        # Win rate: map [30, 70] -> [0, 100]
        norm_wr: float = max(0.0, min(100.0, (avg_win_rate - 30.0) / 40.0 * 100.0))
        # Cost efficiency: already 0-1, map to 0-100
        norm_cost: float = avg_cost_eff * 100.0

        values: list[float] = [
            round(norm_return, 1),
            round(norm_sharpe, 1),
            round(norm_mdd, 1),
            round(norm_wr, 1),
            round(norm_cost, 1),
        ]
        # Close the polygon
        values_closed: list[float] = values + [values[0]]
        categories_closed: list[str] = RADAR_METRICS + [RADAR_METRICS[0]]

        color: str = MODEL_COLORS.get(model, "#888888")

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=DISPLAY_NAMES[model],
                line=dict(color=color, width=2),
                fillcolor=color.replace(")", ", 0.1)").replace("rgb", "rgba")
                if color.startswith("rgb")
                else color + "1A",  # hex alpha ~10% opacity
                hovertemplate=(
                    f"<b>{DISPLAY_NAMES[model]}</b><br>"
                    "%{theta}: %{r:.1f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
            ),
        ),
        title="Model Performance Radar (normalized 0-100)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=500,
        template="plotly_white",
    )

    return fig


# ── Table renderers ─────────────────────────────────────────────


def _render_detailed_matrix_table(matrix_df: pd.DataFrame) -> None:
    """Render a detailed table with all metrics per model x architecture."""
    display_df: pd.DataFrame = matrix_df.copy()
    display_df["Model"] = display_df["model"].map(DISPLAY_NAMES)
    display_df["Arch"] = display_df["architecture"].str.capitalize()

    display_df = display_df.rename(
        columns={
            "total_return_pct": "Return (%)",
            "sharpe_ratio": "Sharpe",
            "max_drawdown_pct": "MDD (%)",
            "win_rate_pct": "Win Rate (%)",
            "cost_efficiency": "Cost Eff.",
            "total_trades": "Trades",
            "avg_confidence": "Avg Conf.",
        }
    )

    display_df = display_df[
        ["Model", "Arch", "Return (%)", "Sharpe", "MDD (%)", "Win Rate (%)", "Cost Eff.", "Trades", "Avg Conf."]
    ].sort_values("Return (%)", ascending=False).reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Return (%)": st.column_config.NumberColumn(format="%.2f"),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "MDD (%)": st.column_config.NumberColumn(format="%.2f"),
            "Win Rate (%)": st.column_config.NumberColumn(format="%.1f"),
            "Cost Eff.": st.column_config.NumberColumn(format="%.3f"),
            "Avg Conf.": st.column_config.NumberColumn(format="%.3f"),
        },
    )


def _render_significance_table(sig_df: pd.DataFrame) -> None:
    """Render statistical significance comparison table."""
    display_df: pd.DataFrame = sig_df.copy()

    # Readable metric names
    metric_labels: dict[str, str] = {
        "total_return_pct": "Total Return",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown_pct": "Max Drawdown",
    }
    display_df["Metric"] = display_df["metric"].map(metric_labels)

    display_df["Significant"] = display_df["significant"].map(
        {True: "Yes (p < 0.05)", False: "No"}
    )

    display_df = display_df.rename(
        columns={
            "model_a": "Model A",
            "model_b": "Model B",
            "p_value": "p-value",
        }
    )

    display_df = display_df[
        ["Model A", "Model B", "Metric", "p-value", "Significant"]
    ].sort_values("p-value").reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "p-value": st.column_config.NumberColumn(format="%.4f"),
        },
    )


# ── Architecture comparison helper ──────────────────────────────


def _build_architecture_comparison(matrix_df: pd.DataFrame) -> go.Figure:
    """Build a grouped bar chart: Single vs Multi for each model, by return."""
    fig: go.Figure = go.Figure()

    for arch in ARCHITECTURES:
        subset: pd.DataFrame = matrix_df[matrix_df["architecture"] == arch]
        subset = subset.set_index("model").reindex(MODELS)

        bar_colors: list[str] = [MODEL_COLORS.get(m, "#888888") for m in MODELS]

        # Use different opacity for single vs multi to distinguish
        opacity: float = 1.0 if arch == ARCH_SINGLE else 0.6

        fig.add_trace(
            go.Bar(
                x=[DISPLAY_NAMES[m] for m in MODELS],
                y=subset["total_return_pct"].tolist(),
                name=arch.capitalize(),
                marker=dict(
                    color=bar_colors,
                    opacity=opacity,
                    line=dict(
                        width=2,
                        color="rgba(0,0,0,0.3)" if arch == ARCH_MULTI else "rgba(0,0,0,0)",
                    ),
                ),
                text=[f"{v:+.2f}%" for v in subset["total_return_pct"]],
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b> (" + arch.capitalize() + ")<br>"
                    "Return: %{y:.2f}%<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Single vs Multi Agent Return by Model",
        xaxis_title="Model",
        yaxis_title="Total Return (%)",
        barmode="group",
        height=400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.3)")

    return fig


# ── Page entry point ────────────────────────────────────────────


def render() -> None:
    """Render the Model Comparison page."""
    st.header("Model Comparison")
    st.caption(
        "4 LLMs x 2 Architectures performance matrix with statistical analysis"
    )

    log.info("rendering_model_comparison_page")

    # Load demo data
    matrix_df: pd.DataFrame = _generate_demo_matrix_data()
    sig_df: pd.DataFrame = _generate_demo_significance()

    # Heatmap
    st.subheader("Performance Heatmap (Return %)")
    fig_heatmap: go.Figure = _build_heatmap(matrix_df)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.divider()

    # Radar chart + Architecture comparison side by side
    col_radar, col_arch = st.columns([1, 1])

    with col_radar:
        st.subheader("Multi-Metric Radar")
        fig_radar: go.Figure = _build_radar_chart(matrix_df)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_arch:
        st.subheader("Architecture Comparison")
        fig_arch: go.Figure = _build_architecture_comparison(matrix_df)
        st.plotly_chart(fig_arch, use_container_width=True)

    st.divider()

    # Detailed table
    st.subheader("Detailed Metrics Table")
    _render_detailed_matrix_table(matrix_df)

    st.divider()

    # Statistical significance
    st.subheader("Statistical Significance (Pairwise)")
    st.caption(
        "Bootstrap-based hypothesis tests (demo data). "
        "p < 0.05 indicates statistically significant difference."
    )
    _render_significance_table(sig_df)
