"""ATLAS Dashboard — Main Streamlit application entry point.

Provides sidebar navigation, auto-refresh (30s), and page routing
for the AI Trading Benchmark dashboard.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import streamlit as st
import structlog

from src.core.logging import setup_logging

# ── Logging ─────────────────────────────────────────────────────
setup_logging(json_output=False)
log: structlog.stdlib.BoundLogger = structlog.get_logger("dashboard.app")

# ── Timezone ────────────────────────────────────────────────────
KST: timezone = timezone(timedelta(hours=9))

# ── Page Registry ───────────────────────────────────────────────
PAGE_OVERVIEW: str = "Overview"
PAGE_MODEL_COMPARISON: str = "Model Comparison"
PAGE_SYSTEM_STATUS: str = "System Status"
PAGE_RISK_DASHBOARD: str = "Risk Dashboard"
PAGE_REGIME_ANALYSIS: str = "Regime Analysis"
PAGE_REPLAY_VIEWER: str = "Replay Viewer"
PAGE_AGENT_DETAIL: str = "Agent Detail"
PAGE_COST_MONITOR: str = "Cost Monitor"
PAGE_MARKET_VIEW: str = "Market View"

PAGES: list[str] = [
    PAGE_OVERVIEW,
    PAGE_MODEL_COMPARISON,
    PAGE_AGENT_DETAIL,
    PAGE_MARKET_VIEW,
    PAGE_COST_MONITOR,
    PAGE_SYSTEM_STATUS,
    PAGE_RISK_DASHBOARD,
    PAGE_REGIME_ANALYSIS,
    PAGE_REPLAY_VIEWER,
]


def _configure_page() -> None:
    """Set Streamlit page config — must be the first st call."""
    st.set_page_config(
        page_title="ATLAS Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _render_sidebar() -> str:
    """Render sidebar with navigation, clock, and refresh controls.

    Returns:
        The selected page name.
    """
    with st.sidebar:
        st.title("ATLAS")
        st.caption("AI Trading Lab for Agent Strategy")

        st.divider()

        selected: str = st.radio(
            label="Navigation",
            options=PAGES,
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Current time in KST
        now_kst: datetime = datetime.now(KST)
        st.metric(
            label="Current Time (KST)",
            value=now_kst.strftime("%H:%M:%S"),
            delta=now_kst.strftime("%Y-%m-%d"),
        )

        # Auto-refresh toggle
        auto_refresh: bool = st.toggle("Auto-refresh (30s)", value=True)
        if auto_refresh:
            st.session_state["auto_refresh"] = True
        else:
            st.session_state["auto_refresh"] = False

        st.divider()
        st.caption("Comparing 4 LLMs x 2 Architectures + Buy&Hold")
        st.caption("Markets: KRX | US | CRYPTO")

    return selected  # type: ignore[return-value]


def _setup_auto_refresh() -> None:
    """Configure 30-second auto-refresh using st.fragment / rerun."""
    if st.session_state.get("auto_refresh", True):
        # Use st.empty placeholder with a countdown approach
        # st_autorefresh from streamlit-autorefresh is optional;
        # fall back to native fragment-based polling.
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore[import-untyped]

            st_autorefresh(interval=30_000, limit=None, key="atlas_autorefresh")
        except ImportError:
            # Graceful degradation: manual refresh only
            log.debug("streamlit_autorefresh not installed; manual refresh only")


def main() -> None:
    """Application entry point."""
    _configure_page()
    _setup_auto_refresh()
    selected_page: str = _render_sidebar()

    log.info("page_selected", page=selected_page)

    if selected_page == PAGE_OVERVIEW:
        from src.dashboard.pages.overview import render

        render()
    elif selected_page == PAGE_MODEL_COMPARISON:
        from src.dashboard.pages.model_comparison import render

        render()
    elif selected_page == PAGE_SYSTEM_STATUS:
        from src.dashboard.pages.system_status import render

        render()
    elif selected_page == PAGE_RISK_DASHBOARD:
        from src.dashboard.pages.risk_dashboard import render

        render()
    elif selected_page == PAGE_REGIME_ANALYSIS:
        from src.dashboard.pages.regime_analysis import render

        render()
    elif selected_page == PAGE_REPLAY_VIEWER:
        from src.dashboard.pages.replay_viewer import render

        render()
    elif selected_page == PAGE_AGENT_DETAIL:
        from src.dashboard.pages.agent_detail import render

        render()
    elif selected_page == PAGE_COST_MONITOR:
        from src.dashboard.pages.cost_monitor import render

        render()
    elif selected_page == PAGE_MARKET_VIEW:
        from src.dashboard.pages.market_view import render

        render()
    else:
        st.error(f"Unknown page: {selected_page}")


if __name__ == "__main__":
    main()
