"""Cost Monitor page — API cost tracking, token breakdown, and cost-adjusted alpha.

Provides:
- Per-model cumulative API cost line chart
- Cost breakdown bar chart (input tokens vs output tokens)
- Cost-Adjusted Alpha (CAA) comparison bar chart
- Cost efficiency table (cost_per_signal)
- Budget utilization progress bars
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from src.core.constants import MODEL_COLORS
from src.core.types import AgentArchitecture, ModelProvider
from src.dashboard.components.charts import (
    CHART_LAYOUT,
    format_kst,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────

_MODELS: list[ModelProvider] = [
    ModelProvider.DEEPSEEK,
    ModelProvider.GEMINI,
    ModelProvider.CLAUDE,
    ModelProvider.GPT,
]
_ARCHS: list[AgentArchitecture] = [
    AgentArchitecture.SINGLE,
    AgentArchitecture.MULTI,
]

_BUDGET_LIMITS: dict[str, float] = {
    ModelProvider.DEEPSEEK.value: 50.0,
    ModelProvider.GEMINI.value: 80.0,
    ModelProvider.CLAUDE.value: 100.0,
    ModelProvider.GPT.value: 60.0,
}

# Pricing per 1k tokens (demo approximations)
_INPUT_COST_PER_1K: dict[str, float] = {
    ModelProvider.DEEPSEEK.value: 0.00014,
    ModelProvider.GEMINI.value: 0.00035,
    ModelProvider.CLAUDE.value: 0.003,
    ModelProvider.GPT.value: 0.00015,
}
_OUTPUT_COST_PER_1K: dict[str, float] = {
    ModelProvider.DEEPSEEK.value: 0.00028,
    ModelProvider.GEMINI.value: 0.00105,
    ModelProvider.CLAUDE.value: 0.015,
    ModelProvider.GPT.value: 0.0006,
}


# ── Demo Data Generators ──────────────────────────────────────────


@st.cache_data(ttl=60)
def _generate_cumulative_costs() -> dict[str, dict[str, list[datetime] | list[float]]]:
    """Generate demo cumulative API cost data per model.

    Returns:
        Dict mapping model label -> {"dates": [...], "costs": [...]}.
    """
    random.seed(101)
    now_utc: datetime = datetime.now(timezone.utc)
    n_points: int = 30

    dates: list[datetime] = [
        now_utc - timedelta(days=n_points - i) for i in range(n_points)
    ]

    result: dict[str, dict[str, list[datetime] | list[float]]] = {}
    for model in _MODELS:
        key: str = model.value
        daily_cost: float = random.uniform(0.5, 4.0)
        cumulative: list[float] = [0.0]
        for _ in range(n_points - 1):
            increment: float = daily_cost * random.uniform(0.6, 1.4)
            cumulative.append(cumulative[-1] + increment)
        result[key] = {"dates": dates, "costs": cumulative}

    return result


@st.cache_data(ttl=60)
def _generate_token_breakdown() -> dict[str, dict[str, float]]:
    """Generate demo input/output token cost breakdown per model.

    Returns:
        Dict mapping model -> {"input_cost": float, "output_cost": float,
        "input_tokens": float, "output_tokens": float}.
    """
    random.seed(102)
    result: dict[str, dict[str, float]] = {}

    for model in _MODELS:
        key: str = model.value
        input_tokens: float = random.uniform(500_000, 5_000_000)
        output_tokens: float = random.uniform(100_000, 2_000_000)
        input_cost: float = (input_tokens / 1000) * _INPUT_COST_PER_1K[key]
        output_cost: float = (output_tokens / 1000) * _OUTPUT_COST_PER_1K[key]

        result[key] = {
            "input_cost": round(input_cost, 2),
            "output_cost": round(output_cost, 2),
            "input_tokens": round(input_tokens),
            "output_tokens": round(output_tokens),
        }

    return result


@st.cache_data(ttl=60)
def _generate_caa_data() -> dict[str, dict[str, float]]:
    """Generate Cost-Adjusted Alpha for each model x architecture combo.

    CAA = (agent_return - buy_hold_return) / total_api_cost

    Returns:
        Dict mapping "model_arch" -> {"agent_return": ..., "bh_return": ...,
        "api_cost": ..., "caa": ...}.
    """
    random.seed(103)
    bh_return: float = round(random.uniform(2.0, 8.0), 2)

    result: dict[str, dict[str, float]] = {}
    for model in _MODELS:
        for arch in _ARCHS:
            label: str = f"{model.value.capitalize()} {arch.value.capitalize()}"
            agent_return: float = round(random.uniform(-2.0, 15.0), 2)
            api_cost: float = round(random.uniform(5.0, 80.0), 2)
            caa: float = round((agent_return - bh_return) / max(api_cost, 0.01), 4)

            result[label] = {
                "agent_return": agent_return,
                "bh_return": bh_return,
                "api_cost": api_cost,
                "caa": caa,
            }

    return result


@st.cache_data(ttl=60)
def _generate_cost_efficiency_table() -> pd.DataFrame:
    """Generate cost-per-signal efficiency table.

    Returns:
        DataFrame with columns: Model, Architecture, Total Signals, Total Cost,
        Cost/Signal, Avg Latency.
    """
    random.seed(104)
    rows: list[dict[str, str | int | float]] = []

    for model in _MODELS:
        for arch in _ARCHS:
            total_signals: int = random.randint(50, 300)
            total_cost: float = round(random.uniform(5.0, 80.0), 2)
            cost_per_signal: float = round(total_cost / max(total_signals, 1), 4)
            avg_latency: float = round(random.uniform(500, 5000), 0)

            rows.append({
                "Model": model.value.capitalize(),
                "Architecture": arch.value.capitalize(),
                "Total Signals": total_signals,
                "Total Cost ($)": total_cost,
                "Cost/Signal ($)": cost_per_signal,
                "Avg Latency (ms)": avg_latency,
            })

    return pd.DataFrame(rows)


# ── Section Renderers ─────────────────────────────────────────────


def _render_cumulative_cost_chart() -> None:
    """Render per-model cumulative API cost line chart."""
    st.subheader("Cumulative API Cost by Model")

    cost_data: dict[str, dict[str, list[datetime] | list[float]]] = (
        _generate_cumulative_costs()
    )

    fig: go.Figure = go.Figure()
    for model_key, series in cost_data.items():
        dates: list[datetime] = series["dates"]  # type: ignore[assignment]
        costs: list[float] = series["costs"]  # type: ignore[assignment]
        kst_dates: list[str] = [format_kst(dt) for dt in dates]
        color: str = MODEL_COLORS.get(model_key, "#FFFFFF")

        fig.add_trace(
            go.Scatter(
                x=kst_dates,
                y=costs,
                mode="lines+markers",
                name=model_key.capitalize(),
                line={"color": color, "width": 2},
                marker={"size": 4},
                hovertemplate="%{x}<br>$%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        title={"text": "Cumulative API Cost Over Time", "x": 0.02, "xanchor": "left"},
        yaxis_title="Cumulative Cost ($)",
        xaxis_title="Time (KST)",
        hovermode="x unified",
        height=450,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_token_breakdown_chart() -> None:
    """Render cost breakdown bar chart (input vs output tokens)."""
    st.subheader("Cost Breakdown: Input vs Output Tokens")

    token_data: dict[str, dict[str, float]] = _generate_token_breakdown()

    models: list[str] = [m.value.capitalize() for m in _MODELS]
    input_costs: list[float] = [
        token_data[m.value]["input_cost"] for m in _MODELS
    ]
    output_costs: list[float] = [
        token_data[m.value]["output_cost"] for m in _MODELS
    ]

    fig: go.Figure = go.Figure()
    fig.add_trace(
        go.Bar(
            x=models,
            y=input_costs,
            name="Input Tokens",
            marker_color="#4285F4",
            text=[f"${c:.2f}" for c in input_costs],
            textposition="inside",
            hovertemplate="Model: %{x}<br>Input Cost: $%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=models,
            y=output_costs,
            name="Output Tokens",
            marker_color="#D97706",
            text=[f"${c:.2f}" for c in output_costs],
            textposition="inside",
            hovertemplate="Model: %{x}<br>Output Cost: $%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Token Cost Breakdown by Model", "x": 0.02, "xanchor": "left"},
        barmode="stack",
        yaxis_title="Cost ($)",
        height=400,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Token count details
    with st.expander("Token Count Details"):
        detail_rows: list[dict[str, str | float]] = []
        for model in _MODELS:
            td: dict[str, float] = token_data[model.value]
            detail_rows.append({
                "Model": model.value.capitalize(),
                "Input Tokens": f"{td['input_tokens']:,.0f}",
                "Output Tokens": f"{td['output_tokens']:,.0f}",
                "Input Cost ($)": f"{td['input_cost']:.2f}",
                "Output Cost ($)": f"{td['output_cost']:.2f}",
                "Total Cost ($)": f"{td['input_cost'] + td['output_cost']:.2f}",
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)


def _render_caa_chart() -> None:
    """Render Cost-Adjusted Alpha comparison bar chart."""
    st.subheader("Cost-Adjusted Alpha (CAA)")
    st.caption("CAA = (Agent Return - Buy & Hold Return) / Total API Cost")

    caa_data: dict[str, dict[str, float]] = _generate_caa_data()

    labels: list[str] = list(caa_data.keys())
    caa_values: list[float] = [v["caa"] for v in caa_data.values()]

    # Colour each bar by model
    bar_colors: list[str] = []
    for label in labels:
        model_name: str = label.split()[0].lower()
        bar_colors.append(MODEL_COLORS.get(model_name, "#FFFFFF"))

    fig: go.Figure = go.Figure(
        data=go.Bar(
            x=labels,
            y=caa_values,
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in caa_values],
            textposition="outside",
            hovertemplate=(
                "Agent: %{x}<br>"
                "CAA: %{y:.4f}<extra></extra>"
            ),
        )
    )

    # Zero-line for reference
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#FAFAFA",
        opacity=0.4,
        annotation_text="Break-even",
        annotation_position="bottom right",
        annotation_font_color="#9E9E9E",
    )

    fig.update_layout(
        title={"text": "Cost-Adjusted Alpha per Agent", "x": 0.02, "xanchor": "left"},
        yaxis_title="CAA (return% / $cost)",
        xaxis_tickangle=-35,
        height=450,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detail table
    with st.expander("CAA Breakdown Details"):
        detail_rows: list[dict[str, str | float]] = []
        for label, vals in caa_data.items():
            detail_rows.append({
                "Agent": label,
                "Agent Return (%)": f"{vals['agent_return']:+.2f}",
                "B&H Return (%)": f"{vals['bh_return']:+.2f}",
                "Alpha (%)": f"{vals['agent_return'] - vals['bh_return']:+.2f}",
                "API Cost ($)": f"{vals['api_cost']:.2f}",
                "CAA": f"{vals['caa']:.4f}",
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)


def _render_cost_efficiency_table() -> None:
    """Render cost-per-signal efficiency table."""
    st.subheader("Cost Efficiency by Agent")

    df: pd.DataFrame = _generate_cost_efficiency_table()
    st.dataframe(
        df.style.highlight_min(  # type: ignore[arg-type]
            subset=["Cost/Signal ($)"],
            color="#22C55E",
            props="color: #0E1117; font-weight: bold",
        ).highlight_max(
            subset=["Cost/Signal ($)"],
            color="#EF4444",
            props="color: #FAFAFA; font-weight: bold",
        ),
        use_container_width=True,
        height=380,
    )


def _render_budget_utilization() -> None:
    """Render budget utilization progress bars per model."""
    st.subheader("Budget Utilization")

    random.seed(105)
    for model in _MODELS:
        key: str = model.value
        budget: float = _BUDGET_LIMITS[key]
        spent: float = round(random.uniform(budget * 0.1, budget * 0.95), 2)
        pct: float = min(spent / budget, 1.0)
        color: str = MODEL_COLORS.get(key, "#FFFFFF")

        col_label, col_bar, col_numbers = st.columns([1, 3, 1])
        with col_label:
            st.markdown(f"**{key.capitalize()}**")
        with col_bar:
            st.progress(pct)
        with col_numbers:
            status_text: str = f"${spent:.2f} / ${budget:.2f}"
            if pct > 0.9:
                st.markdown(f":red[{status_text}]")
            elif pct > 0.7:
                st.markdown(f":orange[{status_text}]")
            else:
                st.markdown(f":green[{status_text}]")


# ── Page Entry Point ──────────────────────────────────────────────


def render() -> None:
    """Render the Cost Monitor page."""
    st.header("Cost Monitor")
    st.caption(f"Updated: {format_kst(datetime.now(timezone.utc))}")

    # ---- Budget overview row ----
    _render_budget_utilization()

    st.divider()

    # ---- Cumulative cost chart ----
    _render_cumulative_cost_chart()

    st.divider()

    # ---- Token breakdown + CAA side by side ----
    col_tokens, col_caa = st.columns(2)
    with col_tokens:
        _render_token_breakdown_chart()
    with col_caa:
        _render_caa_chart()

    st.divider()

    # ---- Efficiency table ----
    _render_cost_efficiency_table()

    logger.debug("cost_monitor_rendered")
