"""Risk management dashboard — real-time risk monitoring and alerts."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.core.logging import get_logger

log = get_logger(__name__)


def render() -> None:
    """Render the risk management dashboard page."""
    st.header("Risk Management Dashboard")

    # ── Portfolio Risk Overview ─────────────────────────────────
    st.subheader("Portfolio Risk Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Portfolio VaR (95%)", "2.3%", "-0.1%")
    with col2:
        st.metric("Max Drawdown", "8.2%", "+1.5%")
    with col3:
        st.metric("Current Drawdown", "3.1%", "-0.8%")
    with col4:
        st.metric("Daily Loss", "0.4%", "0.0%")

    # ── Risk Check Results ─────────────────────────────────────
    st.subheader("Risk Check Results (Last Cycle)")

    risk_checks = [
        {"Check": "Position Limit", "Status": "PASS", "Value": "22.0%", "Limit": "30.0%"},
        {"Check": "Cash Reserve", "Status": "PASS", "Value": "35.2%", "Limit": "20.0%"},
        {"Check": "Confidence Threshold", "Status": "PASS", "Value": "0.75", "Limit": "0.30"},
        {"Check": "Drawdown Circuit Breaker", "Status": "PASS", "Value": "3.1%", "Limit": "15.0%"},
        {"Check": "Daily Loss Limit", "Status": "PASS", "Value": "0.4%", "Limit": "5.0%"},
    ]
    st.dataframe(risk_checks, use_container_width=True)

    # ── VaR Time Series ────────────────────────────────────────
    st.subheader("Value-at-Risk Over Time")
    st.info(
        "Connect to live benchmark data to see real-time VaR charts. "
        "This page displays placeholder structure."
    )

    fig_var = go.Figure()
    fig_var.add_trace(go.Scatter(
        x=list(range(30)),
        y=[2.0 + 0.1 * i % 5 for i in range(30)],
        mode="lines",
        name="VaR 95%",
        line=dict(color="red"),
    ))
    fig_var.add_trace(go.Scatter(
        x=list(range(30)),
        y=[3.5 + 0.15 * i % 7 for i in range(30)],
        mode="lines",
        name="VaR 99%",
        line=dict(color="darkred", dash="dash"),
    ))
    fig_var.update_layout(
        title="Historical VaR",
        xaxis_title="Trading Day",
        yaxis_title="VaR (%)",
        template="plotly_dark",
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # ── Drawdown Chart ─────────────────────────────────────────
    st.subheader("Drawdown Analysis")
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=list(range(30)),
        y=[-1.0 - 0.2 * i % 8 for i in range(30)],
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        line=dict(color="crimson"),
    ))
    fig_dd.add_hline(y=-15.0, line_dash="dash", line_color="yellow",
                      annotation_text="Circuit Breaker (-15%)")
    fig_dd.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Trading Day",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Position Concentration ─────────────────────────────────
    st.subheader("Position Concentration")
    col1, col2 = st.columns(2)

    with col1:
        fig_conc = px.pie(
            values=[22, 18, 15, 10, 35],
            names=["BTCUSDT", "ETHUSDT", "AAPL", "005930", "Cash"],
            title="Portfolio Allocation",
        )
        fig_conc.update_layout(template="plotly_dark")
        st.plotly_chart(fig_conc, use_container_width=True)

    with col2:
        st.markdown("**Position Limits**")
        st.progress(0.73, text="BTCUSDT: 22.0% / 30.0%")
        st.progress(0.60, text="ETHUSDT: 18.0% / 30.0%")
        st.progress(0.50, text="AAPL: 15.0% / 30.0%")
        st.progress(0.33, text="005930: 10.0% / 30.0%")

    # ── Volatility Regime ──────────────────────────────────────
    st.subheader("Volatility Regime Monitor")
    regime_col1, regime_col2 = st.columns(2)
    with regime_col1:
        st.metric("Current Regime", "NORMAL")
        st.metric("Annualised Volatility", "18.3%")
    with regime_col2:
        st.metric("Regime Duration", "5 days")
        st.metric("Previous Regime", "HIGH")

    # ── Risk Alerts ────────────────────────────────────────────
    st.subheader("Risk Alerts")
    st.info("No active risk alerts. All checks passing.")

    log.debug("risk_dashboard_rendered")
