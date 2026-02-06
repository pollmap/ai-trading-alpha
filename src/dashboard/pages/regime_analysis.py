"""Regime analysis dashboard — market regime detection and performance by regime."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.core.logging import get_logger

log = get_logger(__name__)


def render() -> None:
    """Render the regime analysis page."""
    st.header("Market Regime Analysis")

    # ── Current Regime ─────────────────────────────────────────
    st.subheader("Current Market Regime")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("KRX Regime", "SIDEWAYS")
        st.caption("SMA50 near SMA200, low volatility")
    with col2:
        st.metric("US Regime", "BULL")
        st.caption("SMA50 > SMA200, positive momentum")
    with col3:
        st.metric("Crypto Regime", "HIGH_VOLATILITY")
        st.caption("ATR elevated, mixed signals")

    # ── Regime History ─────────────────────────────────────────
    st.subheader("Regime History")
    st.info(
        "Connect to live benchmark data for real-time regime tracking. "
        "Showing placeholder structure."
    )

    # Regime timeline visualization
    fig_timeline = go.Figure()

    regimes = ["BULL", "BULL", "SIDEWAYS", "SIDEWAYS", "BEAR",
                "BEAR", "BEAR", "SIDEWAYS", "BULL", "BULL"]
    colors = {
        "BULL": "green", "BEAR": "red",
        "SIDEWAYS": "gray", "HIGH_VOLATILITY": "orange",
        "CRASH": "darkred",
    }
    regime_colors = [colors.get(r, "blue") for r in regimes]

    fig_timeline.add_trace(go.Bar(
        x=list(range(len(regimes))),
        y=[1] * len(regimes),
        marker_color=regime_colors,
        text=regimes,
        textposition="inside",
        name="Regime",
    ))
    fig_timeline.update_layout(
        title="Regime Timeline (US Market)",
        xaxis_title="Week",
        yaxis_visible=False,
        template="plotly_dark",
        showlegend=False,
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Performance by Regime ──────────────────────────────────
    st.subheader("Agent Performance by Regime")

    perf_data = {
        "Model": ["DeepSeek", "Gemini", "Claude", "GPT-4o-mini"] * 3,
        "Regime": ["BULL"] * 4 + ["SIDEWAYS"] * 4 + ["BEAR"] * 4,
        "Return (%)": [8.2, 7.5, 9.1, 6.8, 1.2, 0.8, 2.1, -0.5,
                        -3.1, -2.5, -1.8, -4.2],
    }

    fig_perf = px.bar(
        perf_data,
        x="Model",
        y="Return (%)",
        color="Regime",
        barmode="group",
        title="Model Returns by Market Regime",
        color_discrete_map={"BULL": "green", "SIDEWAYS": "gray", "BEAR": "red"},
    )
    fig_perf.update_layout(template="plotly_dark")
    st.plotly_chart(fig_perf, use_container_width=True)

    # ── Regime Detection Signals ───────────────────────────────
    st.subheader("Regime Detection Indicators")

    col1, col2 = st.columns(2)

    with col1:
        fig_sma = go.Figure()
        sma50 = [100 + i * 0.5 for i in range(50)]
        sma200 = [95 + i * 0.3 for i in range(50)]
        fig_sma.add_trace(go.Scatter(
            y=sma50, mode="lines", name="SMA 50", line=dict(color="blue"),
        ))
        fig_sma.add_trace(go.Scatter(
            y=sma200, mode="lines", name="SMA 200", line=dict(color="orange"),
        ))
        fig_sma.update_layout(
            title="SMA Crossover (Regime Signal)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_sma, use_container_width=True)

    with col2:
        fig_vol = go.Figure()
        vol_data = [15 + 3 * (i % 7) for i in range(50)]
        fig_vol.add_trace(go.Scatter(
            y=vol_data, mode="lines", name="Annualised Vol",
            line=dict(color="purple"),
        ))
        fig_vol.add_hline(y=25, line_dash="dash", line_color="orange",
                           annotation_text="HIGH threshold")
        fig_vol.add_hline(y=50, line_dash="dash", line_color="red",
                           annotation_text="EXTREME threshold")
        fig_vol.update_layout(
            title="Volatility (Regime Signal)",
            template="plotly_dark",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── Regime Transition Matrix ───────────────────────────────
    st.subheader("Regime Transition Probabilities")

    transition = [
        [0.7, 0.2, 0.08, 0.02],
        [0.15, 0.6, 0.2, 0.05],
        [0.05, 0.25, 0.6, 0.1],
        [0.1, 0.15, 0.25, 0.5],
    ]
    labels = ["BULL", "SIDEWAYS", "BEAR", "HIGH_VOL"]

    fig_trans = go.Figure(data=go.Heatmap(
        z=transition,
        x=labels,
        y=labels,
        colorscale="RdYlGn",
        text=[[f"{v:.0%}" for v in row] for row in transition],
        texttemplate="%{text}",
    ))
    fig_trans.update_layout(
        title="Regime Transition Matrix",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        template="plotly_dark",
    )
    st.plotly_chart(fig_trans, use_container_width=True)

    log.debug("regime_analysis_rendered")
