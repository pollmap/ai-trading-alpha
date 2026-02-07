"""System Status page — benchmark progress, agent health, error log, countdown.

Shows real-time operational health of the ATLAS benchmark system.
Uses demo data when the database is not connected.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

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

log: structlog.stdlib.BoundLogger = structlog.get_logger("dashboard.system_status")

# ── Timezone ────────────────────────────────────────────────────
KST: timezone = timezone(timedelta(hours=9))

# ── Constants ───────────────────────────────────────────────────
CYCLE_INTERVAL_MINUTES: int = 30
TOTAL_PLANNED_CYCLES: int = 100


class AgentStatus(str, Enum):
    """Health status of an individual agent."""

    RUNNING = "running"
    IDLE = "idle"
    FAILED = "failed"
    COMPLETED = "completed"


# ── Demo data generation ────────────────────────────────────────

AGENTS: list[tuple[str, str]] = [
    (MODEL_DEEPSEEK, ARCH_SINGLE),
    (MODEL_DEEPSEEK, ARCH_MULTI),
    (MODEL_GEMINI, ARCH_SINGLE),
    (MODEL_GEMINI, ARCH_MULTI),
    (MODEL_CLAUDE, ARCH_SINGLE),
    (MODEL_CLAUDE, ARCH_MULTI),
    (MODEL_GPT, ARCH_SINGLE),
    (MODEL_GPT, ARCH_MULTI),
]


@st.cache_data(ttl=60)
def _generate_demo_progress() -> dict[str, int]:
    """Generate demo benchmark progress data.

    Returns:
        Dict with completed_cycles, total_cycles, errors_count.
    """
    return {
        "completed_cycles": 47,
        "total_cycles": TOTAL_PLANNED_CYCLES,
        "errors_count": 3,
        "start_time_utc": int(
            (datetime.now(timezone.utc) - timedelta(days=2, hours=5)).timestamp()
        ),
    }


@st.cache_data(ttl=60)
def _generate_demo_agent_health() -> pd.DataFrame:
    """Generate demo agent health status for all 8 agents.

    Returns:
        DataFrame with columns: agent, model, architecture, status,
        last_cycle_utc, cycles_completed, consecutive_errors.
    """
    statuses: list[str] = [
        AgentStatus.RUNNING,
        AgentStatus.RUNNING,
        AgentStatus.RUNNING,
        AgentStatus.IDLE,
        AgentStatus.RUNNING,
        AgentStatus.RUNNING,
        AgentStatus.FAILED,
        AgentStatus.RUNNING,
    ]

    now_utc: datetime = datetime.now(timezone.utc)
    rows: list[dict[str, object]] = []

    for idx, (model, arch) in enumerate(AGENTS):
        label: str = f"{model.capitalize()} ({arch.capitalize()})"
        status: str = statuses[idx]
        minutes_ago: int = idx * 3 + 1
        last_cycle: datetime = now_utc - timedelta(minutes=minutes_ago)
        cycles_done: int = 47 - (idx % 3)
        consec_errors: int = 3 if status == AgentStatus.FAILED else 0

        rows.append(
            {
                "agent": label,
                "model": model,
                "architecture": arch,
                "status": status,
                "last_cycle_utc": last_cycle,
                "cycles_completed": cycles_done,
                "consecutive_errors": consec_errors,
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def _generate_demo_error_log() -> pd.DataFrame:
    """Generate demo error log entries.

    Returns:
        DataFrame with columns: timestamp_utc, agent, error_type, message.
    """
    now_utc: datetime = datetime.now(timezone.utc)
    errors: list[dict[str, object]] = [
        {
            "timestamp_utc": now_utc - timedelta(hours=2, minutes=15),
            "agent": "GPT (Multi)",
            "error_type": "LLM_PARSE_ERROR",
            "message": "Failed to parse JSON response from GPT-4o-mini; returned HOLD as fallback.",
        },
        {
            "timestamp_utc": now_utc - timedelta(hours=5, minutes=42),
            "agent": "DeepSeek (Single)",
            "error_type": "TIMEOUT",
            "message": "LLM call exceeded 30s timeout. Retried successfully on attempt 2.",
        },
        {
            "timestamp_utc": now_utc - timedelta(days=1, hours=1),
            "agent": "GPT (Multi)",
            "error_type": "API_ERROR",
            "message": "OpenAI API returned 429 Too Many Requests. Backed off 60s.",
        },
        {
            "timestamp_utc": now_utc - timedelta(days=1, hours=8),
            "agent": "Gemini (Multi)",
            "error_type": "LLM_PARSE_ERROR",
            "message": "Gemini response missing 'action' field; returned HOLD as fallback.",
        },
        {
            "timestamp_utc": now_utc - timedelta(days=1, hours=12),
            "agent": "GPT (Multi)",
            "error_type": "TIMEOUT",
            "message": "Multi-agent debate timed out at 120s. Used last available signal.",
        },
    ]
    return pd.DataFrame(errors)


# ── Chart builders ──────────────────────────────────────────────


def _build_progress_gauge(completed: int, total: int) -> go.Figure:
    """Build a gauge chart showing benchmark completion percentage."""
    pct: float = (completed / total * 100) if total > 0 else 0.0

    fig: go.Figure = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 40}},
            delta={"reference": 100, "relative": False, "position": "bottom"},
            title={"text": "Benchmark Progress", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#4285F4"},
                "steps": [
                    {"range": [0, 33], "color": "#FFECB3"},
                    {"range": [33, 66], "color": "#C8E6C9"},
                    {"range": [66, 100], "color": "#B3E5FC"},
                ],
                "threshold": {
                    "line": {"color": "#D32F2F", "width": 3},
                    "thickness": 0.8,
                    "value": 100,
                },
            },
        )
    )

    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
    return fig


def _build_agent_status_chart(health_df: pd.DataFrame) -> go.Figure:
    """Build a horizontal bar chart visualizing agent health status."""
    status_colors: dict[str, str] = {
        AgentStatus.RUNNING: "#4CAF50",
        AgentStatus.IDLE: "#FFC107",
        AgentStatus.FAILED: "#F44336",
        AgentStatus.COMPLETED: "#2196F3",
    }

    fig: go.Figure = go.Figure()

    for status in [AgentStatus.RUNNING, AgentStatus.IDLE, AgentStatus.FAILED, AgentStatus.COMPLETED]:
        subset: pd.DataFrame = health_df[health_df["status"] == status]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                y=subset["agent"],
                x=subset["cycles_completed"],
                orientation="h",
                name=status.value.capitalize(),
                marker_color=status_colors[status],
                text=subset["cycles_completed"],
                textposition="auto",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Cycles: %{x}<br>"
                    f"Status: {status.value}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Agent Cycles Completed (colored by status)",
        xaxis_title="Cycles Completed",
        yaxis_title="",
        barmode="stack",
        height=350,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=150, b=80),
    )

    return fig


# ── Page sections ───────────────────────────────────────────────


def _render_progress_section(progress: dict[str, int]) -> None:
    """Render the benchmark progress section with gauge and stats."""
    col_gauge, col_stats = st.columns([1, 1])

    completed: int = progress["completed_cycles"]
    total: int = progress["total_cycles"]
    errors: int = progress["errors_count"]

    with col_gauge:
        fig: go.Figure = _build_progress_gauge(completed, total)
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.metric("Completed Cycles", f"{completed} / {total}")
        st.metric("Remaining Cycles", f"{total - completed}")
        st.metric("Total Errors", f"{errors}")

        # Estimated time remaining
        if completed > 0:
            start_ts: int = progress["start_time_utc"]
            start_dt: datetime = datetime.fromtimestamp(start_ts, tz=timezone.utc)
            elapsed: timedelta = datetime.now(timezone.utc) - start_dt
            avg_cycle_time: timedelta = elapsed / completed
            remaining_time: timedelta = avg_cycle_time * (total - completed)
            eta_utc: datetime = datetime.now(timezone.utc) + remaining_time
            eta_kst: datetime = eta_utc.astimezone(KST)
            st.metric("Estimated Completion (KST)", eta_kst.strftime("%Y-%m-%d %H:%M"))


def _render_agent_health(health_df: pd.DataFrame) -> None:
    """Render agent health status table and chart."""
    col_chart, col_table = st.columns([1, 1])

    with col_chart:
        fig: go.Figure = _build_agent_status_chart(health_df)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        display_df: pd.DataFrame = health_df.copy()

        # Convert last_cycle to KST string
        display_df["last_cycle_kst"] = display_df["last_cycle_utc"].apply(
            lambda dt: dt.astimezone(KST).strftime("%m/%d %H:%M:%S")
            if isinstance(dt, datetime)
            else str(dt)
        )

        # Status badge using emoji-free indicators
        status_map: dict[str, str] = {
            AgentStatus.RUNNING: "RUNNING",
            AgentStatus.IDLE: "IDLE",
            AgentStatus.FAILED: "FAILED",
            AgentStatus.COMPLETED: "DONE",
        }
        display_df["status_label"] = display_df["status"].map(status_map)

        st.dataframe(
            display_df[
                [
                    "agent",
                    "status_label",
                    "cycles_completed",
                    "consecutive_errors",
                    "last_cycle_kst",
                ]
            ].rename(
                columns={
                    "agent": "Agent",
                    "status_label": "Status",
                    "cycles_completed": "Cycles",
                    "consecutive_errors": "Consec. Errors",
                    "last_cycle_kst": "Last Cycle (KST)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_error_log(error_df: pd.DataFrame) -> None:
    """Render the error log as a table."""
    display_df: pd.DataFrame = error_df.copy()

    # Convert timestamps to KST
    display_df["timestamp_kst"] = display_df["timestamp_utc"].apply(
        lambda dt: dt.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(dt, datetime)
        else str(dt)
    )

    display_df = display_df.sort_values("timestamp_utc", ascending=False).reset_index(
        drop=True
    )

    st.dataframe(
        display_df[["timestamp_kst", "agent", "error_type", "message"]].rename(
            columns={
                "timestamp_kst": "Time (KST)",
                "agent": "Agent",
                "error_type": "Error Type",
                "message": "Message",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_next_cycle_countdown() -> None:
    """Render a countdown to the next benchmark cycle."""
    now_utc: datetime = datetime.now(timezone.utc)

    # Simulate: next cycle is at the next 30-minute boundary
    minutes_since_midnight: int = now_utc.hour * 60 + now_utc.minute
    next_boundary: int = (
        (minutes_since_midnight // CYCLE_INTERVAL_MINUTES) + 1
    ) * CYCLE_INTERVAL_MINUTES
    next_cycle_utc: datetime = now_utc.replace(
        hour=0, minute=0, second=0, microsecond=0
    ) + timedelta(minutes=next_boundary)
    remaining: timedelta = next_cycle_utc - now_utc

    minutes_left: int = int(remaining.total_seconds() // 60)
    seconds_left: int = int(remaining.total_seconds() % 60)

    next_kst: datetime = next_cycle_utc.astimezone(KST)

    col_time, col_countdown = st.columns(2)
    with col_time:
        st.metric(
            label="Next Cycle (KST)",
            value=next_kst.strftime("%H:%M:%S"),
        )
    with col_countdown:
        st.metric(
            label="Time Remaining",
            value=f"{minutes_left}m {seconds_left}s",
        )


# ── Page entry point ────────────────────────────────────────────


def render() -> None:
    """Render the System Status page."""
    st.header("System Status")
    st.caption("Real-time operational health of the ATLAS benchmark")

    log.info("rendering_system_status_page")

    # Next cycle countdown
    st.subheader("Next Cycle")
    _render_next_cycle_countdown()

    st.divider()

    # Benchmark progress
    st.subheader("Benchmark Progress")
    progress: dict[str, int] = _generate_demo_progress()
    _render_progress_section(progress)

    st.divider()

    # Agent health
    st.subheader("Agent Health")
    health_df: pd.DataFrame = _generate_demo_agent_health()
    _render_agent_health(health_df)

    st.divider()

    # Error log
    st.subheader("Recent Errors")
    error_df: pd.DataFrame = _generate_demo_error_log()
    _render_error_log(error_df)
