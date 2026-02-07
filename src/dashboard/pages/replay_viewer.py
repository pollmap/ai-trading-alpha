"""Replay viewer dashboard — browse and inspect recorded trading sessions."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from src.core.logging import get_logger

log = get_logger(__name__)


def render() -> None:
    """Render the replay viewer page."""
    st.header("Session Replay Viewer")

    # ── Session Selector ───────────────────────────────────────
    st.subheader("Select Session")

    sessions = [
        "benchmark-2026-01-15-full",
        "benchmark-2026-01-14-full",
        "benchmark-2026-01-13-partial",
    ]
    selected = st.selectbox("Replay Session", sessions)
    st.caption(f"Selected: {selected}")

    # ── Session Overview ───────────────────────────────────────
    st.subheader("Session Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Events", "1,247")
    with col2:
        st.metric("Snapshots", "48")
    with col3:
        st.metric("Signals Generated", "384")
    with col4:
        st.metric("Trades Executed", "156")

    # ── Event Timeline ─────────────────────────────────────────
    st.subheader("Event Timeline")

    event_types = ["snapshot", "signal", "risk_check", "trade", "portfolio_update"]
    event_filter = st.multiselect(
        "Filter Event Types",
        event_types,
        default=["signal", "trade"],
    )

    st.info(
        "Load a replay file to see the full event timeline. "
        "Showing placeholder structure."
    )

    # Placeholder timeline
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=list(range(20)),
        y=[1, 2, 1, 3, 1, 2, 3, 1, 2, 1, 3, 2, 1, 2, 3, 1, 2, 1, 3, 2],
        mode="markers+lines",
        marker=dict(
            size=10,
            color=["blue", "green", "blue", "red", "blue",
                   "green", "red", "blue", "green", "blue",
                   "red", "green", "blue", "green", "red",
                   "blue", "green", "blue", "red", "green"],
        ),
        text=["SNP", "SIG", "SNP", "TRD", "SNP",
              "SIG", "TRD", "SNP", "SIG", "SNP",
              "TRD", "SIG", "SNP", "SIG", "TRD",
              "SNP", "SIG", "SNP", "TRD", "SIG"],
        name="Events",
    ))
    fig_timeline.update_layout(
        title="Event Sequence",
        xaxis_title="Sequence #",
        yaxis_title="Event Type",
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=["Snapshot", "Signal", "Trade"],
        ),
        template="plotly_dark",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── Event Inspector ────────────────────────────────────────
    st.subheader("Event Inspector")

    event_num = st.number_input("Event #", min_value=1, max_value=100, value=1)
    st.json({
        "event_type": "signal",
        "timestamp": "2026-01-15T09:30:00+00:00",
        "sequence_number": event_num,
        "data": {
            "signal_id": "01EXAMPLE",
            "symbol": "BTCUSDT",
            "action": "BUY",
            "weight": 0.15,
            "confidence": 0.82,
            "reasoning": "Strong momentum with bullish MACD crossover...",
            "model": "deepseek",
            "architecture": "single",
        },
    })

    # ── Decision Trail ─────────────────────────────────────────
    st.subheader("Decision Trail")
    st.markdown("""
    **Signal → Risk Check → Trade → Portfolio Update**

    1. **Snapshot #23**: Market data collected (BTCUSDT: $43,250)
    2. **Signal #45**: DeepSeek/Single → BUY BTCUSDT (weight=0.15, conf=0.82)
    3. **Risk Check**: All 5 checks PASSED
    4. **Trade #67**: BUY 0.034 BTC @ $43,293 (commission: $1.30)
    5. **Portfolio**: Cash $96,520 → Position BTCUSDT 0.034 BTC
    """)

    # ── Comparison Mode ────────────────────────────────────────
    st.subheader("Side-by-Side Comparison")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown("**DeepSeek / Single**")
        st.markdown("- Signal: BUY (conf=0.82)")
        st.markdown("- Risk: APPROVED")
        st.markdown("- Trade: +0.034 BTC")
        st.markdown("- PnL: +$125.30")

    with comp_col2:
        st.markdown("**Gemini / Single**")
        st.markdown("- Signal: HOLD (conf=0.65)")
        st.markdown("- Risk: N/A")
        st.markdown("- Trade: None")
        st.markdown("- PnL: $0.00")

    log.debug("replay_viewer_rendered")
