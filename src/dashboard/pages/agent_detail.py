"""Agent Detail page — deep-dive into a single model + architecture combination.

Provides:
- Dropdown selector for model x architecture
- Decision timeline scatter (BUY / SELL / HOLD)
- Reasoning text display per decision
- Action distribution pie chart
- Win rate and profit factor metrics
- Confidence histogram
- Agent personality profile (contrarian score, avg confidence, preferred action)
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import plotly.graph_objects as go
import streamlit as st
import structlog

from src.core.constants import MODEL_COLORS
from src.core.types import Action, AgentArchitecture, ModelProvider
from src.dashboard.components.charts import (
    CHART_LAYOUT,
    create_radar_chart,
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

_ACTION_COLORS: dict[str, str] = {
    Action.BUY.value: "#22C55E",
    Action.SELL.value: "#EF4444",
    Action.HOLD.value: "#9E9E9E",
}

_DEMO_SYMBOLS: list[str] = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

_REASONING_TEMPLATES: dict[str, list[str]] = {
    Action.BUY.value: [
        "Strong upward momentum detected. RSI recovering from oversold territory. "
        "Volume confirms breakout above 20-day MA. Fundamentals support accumulation.",
        "Sector rotation favoring tech. Positive earnings surprise expected based on "
        "supply chain data. Risk-reward ratio attractive at current levels.",
        "Macro tailwinds from dovish Fed commentary. Options flow showing heavy call buying. "
        "Technical support held firmly on recent pullback.",
    ],
    Action.SELL.value: [
        "Bearish divergence on RSI with price making new highs. Volume declining on rallies. "
        "Approaching key resistance with deteriorating breadth.",
        "Negative news catalyst imminent based on sentiment analysis. Position sizing warrants "
        "risk reduction. Trailing stop triggered on intraday weakness.",
        "Overvaluation concerns at current PER levels. Institutional selling pressure detected "
        "in dark pool data. Rotating to defensive posture.",
    ],
    Action.HOLD.value: [
        "Mixed signals across timeframes. Waiting for confirmation of breakout direction. "
        "Current position size appropriate for volatility regime.",
        "Market in consolidation phase. No clear edge detected in current price action. "
        "Preserving capital for higher-conviction opportunities.",
        "Conflicting macro signals: strong employment but rising yields. "
        "Maintaining current exposure until data clarifies trend.",
    ],
}


# ── Demo Data Generators ──────────────────────────────────────────


@st.cache_data(ttl=60)
def _generate_decision_timeline(
    model_key: str,
    arch_key: str,
) -> list[dict[str, str | float | datetime]]:
    """Generate a timeline of demo trading decisions.

    Args:
        model_key: ModelProvider value.
        arch_key: AgentArchitecture value.

    Returns:
        List of decision dicts with keys: timestamp, symbol, action,
        confidence, reasoning.
    """
    random.seed(hash(f"{model_key}_{arch_key}_timeline") % (2**31))
    now_utc: datetime = datetime.now(timezone.utc)
    n_decisions: int = 40

    actions_list: list[str] = [Action.BUY.value, Action.SELL.value, Action.HOLD.value]
    weights: list[float] = [0.3, 0.2, 0.5]

    decisions: list[dict[str, str | float | datetime]] = []
    for i in range(n_decisions):
        ts: datetime = now_utc - timedelta(hours=(n_decisions - i) * 6)
        action: str = random.choices(actions_list, weights=weights, k=1)[0]
        confidence: float = round(random.uniform(0.3, 0.95), 2)
        symbol: str = random.choice(_DEMO_SYMBOLS)
        reasoning: str = random.choice(_REASONING_TEMPLATES[action])

        decisions.append({
            "timestamp": ts,
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
        })

    return decisions


@st.cache_data(ttl=60)
def _generate_performance_metrics(
    model_key: str,
    arch_key: str,
) -> dict[str, float]:
    """Generate demo performance metrics for an agent.

    Args:
        model_key: ModelProvider value.
        arch_key: AgentArchitecture value.

    Returns:
        Dict with metrics: win_rate, profit_factor, total_trades, avg_return,
        max_drawdown, sharpe_ratio.
    """
    random.seed(hash(f"{model_key}_{arch_key}_metrics") % (2**31))
    return {
        "win_rate": round(random.uniform(0.40, 0.65), 3),
        "profit_factor": round(random.uniform(0.8, 2.5), 2),
        "total_trades": random.randint(20, 80),
        "avg_return": round(random.uniform(-0.5, 1.5), 2),
        "max_drawdown": round(random.uniform(-15, -3), 2),
        "sharpe_ratio": round(random.uniform(-0.5, 2.0), 2),
    }


@st.cache_data(ttl=60)
def _generate_personality_profile(
    model_key: str,
    arch_key: str,
) -> dict[str, float | str]:
    """Generate agent personality profile.

    Args:
        model_key: ModelProvider value.
        arch_key: AgentArchitecture value.

    Returns:
        Dict with: contrarian_score, avg_confidence, preferred_action,
        risk_appetite, decisiveness, consistency.
    """
    random.seed(hash(f"{model_key}_{arch_key}_personality") % (2**31))

    actions: list[str] = [Action.BUY.value, Action.SELL.value, Action.HOLD.value]
    preferred: str = random.choice(actions)

    return {
        "contrarian_score": round(random.uniform(10, 90), 1),
        "avg_confidence": round(random.uniform(0.4, 0.85), 2),
        "preferred_action": preferred,
        "risk_appetite": round(random.uniform(20, 90), 1),
        "decisiveness": round(random.uniform(30, 95), 1),
        "consistency": round(random.uniform(40, 90), 1),
    }


# ── Section Renderers ─────────────────────────────────────────────


def _render_decision_timeline(
    decisions: list[dict[str, str | float | datetime]],
) -> None:
    """Render the decision scatter-plot timeline.

    Args:
        decisions: List of decision dicts.
    """
    st.subheader("Decision Timeline")

    fig: go.Figure = go.Figure()

    for action_val in [Action.BUY.value, Action.SELL.value, Action.HOLD.value]:
        filtered: list[dict[str, str | float | datetime]] = [
            d for d in decisions if d["action"] == action_val
        ]
        if not filtered:
            continue

        timestamps: list[str] = [
            format_kst(d["timestamp"])  # type: ignore[arg-type]
            for d in filtered
        ]
        confidences: list[float] = [
            float(d["confidence"]) for d in filtered
        ]
        symbols: list[str] = [str(d["symbol"]) for d in filtered]
        hover_texts: list[str] = [
            f"Symbol: {d['symbol']}<br>Confidence: {d['confidence']:.0%}"
            for d in filtered
        ]

        marker_symbol: str = (
            "triangle-up" if action_val == Action.BUY.value
            else "triangle-down" if action_val == Action.SELL.value
            else "circle"
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=confidences,
                mode="markers",
                name=action_val,
                marker={
                    "color": _ACTION_COLORS[action_val],
                    "size": 10,
                    "symbol": marker_symbol,
                    "line": {"width": 1, "color": "#FFFFFF"},
                },
                text=hover_texts,
                hovertemplate="%{text}<br>Time: %{x}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        title={"text": "Trading Decisions Over Time", "x": 0.02, "xanchor": "left"},
        yaxis_title="Confidence",
        xaxis_title="Time (KST)",
        yaxis_range=[0, 1.05],
        hovermode="closest",
        height=450,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_reasoning_browser(
    decisions: list[dict[str, str | float | datetime]],
) -> None:
    """Render a browseable list of decision reasoning texts.

    Args:
        decisions: List of decision dicts.
    """
    st.subheader("Decision Reasoning")

    # Show most recent decisions first
    sorted_decisions: list[dict[str, str | float | datetime]] = sorted(
        decisions, key=lambda d: d["timestamp"], reverse=True,  # type: ignore[arg-type,return-value]
    )

    n_display: int = st.slider(
        "Number of decisions to show",
        min_value=5,
        max_value=min(len(sorted_decisions), 40),
        value=10,
        key="reasoning_slider",
    )

    for decision in sorted_decisions[:n_display]:
        action_str: str = str(decision["action"])
        symbol_str: str = str(decision["symbol"])
        confidence_val: float = float(decision["confidence"])
        timestamp_dt: datetime = decision["timestamp"]  # type: ignore[assignment]
        reasoning_str: str = str(decision["reasoning"])

        color: str = _ACTION_COLORS.get(action_str, "#FFFFFF")
        kst_str: str = format_kst(timestamp_dt)

        with st.expander(
            f":{color[1:]}[**{action_str}**] {symbol_str} — "
            f"Confidence: {confidence_val:.0%} — {kst_str}",
            expanded=False,
        ):
            st.markdown(f"> {reasoning_str}")


def _render_action_distribution(
    decisions: list[dict[str, str | float | datetime]],
) -> None:
    """Render a pie chart of action distribution.

    Args:
        decisions: List of decision dicts.
    """
    st.subheader("Action Distribution")

    action_counts: dict[str, int] = {a.value: 0 for a in Action}
    for d in decisions:
        action_str: str = str(d["action"])
        action_counts[action_str] = action_counts.get(action_str, 0) + 1

    labels: list[str] = list(action_counts.keys())
    values: list[int] = list(action_counts.values())
    colors: list[str] = [_ACTION_COLORS.get(l, "#FFFFFF") for l in labels]

    fig: go.Figure = go.Figure(
        data=go.Pie(
            labels=labels,
            values=values,
            marker={"colors": colors},
            hole=0.45,
            textinfo="label+percent",
            textfont={"size": 13, "color": "#FAFAFA"},
            hovertemplate="%{label}: %{value} decisions (%{percent})<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Action Distribution", "x": 0.02, "xanchor": "left"},
        height=380,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_confidence_histogram(
    decisions: list[dict[str, str | float | datetime]],
) -> None:
    """Render a histogram of confidence scores.

    Args:
        decisions: List of decision dicts.
    """
    st.subheader("Confidence Distribution")

    confidences: list[float] = [float(d["confidence"]) for d in decisions]

    fig: go.Figure = go.Figure(
        data=go.Histogram(
            x=confidences,
            nbinsx=20,
            marker_color="#4285F4",
            opacity=0.85,
            hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title={"text": "Confidence Score Distribution", "x": 0.02, "xanchor": "left"},
        xaxis_title="Confidence",
        yaxis_title="Count",
        xaxis_range=[0, 1],
        height=350,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics(metrics: dict[str, float]) -> None:
    """Render key performance metrics as metric cards.

    Args:
        metrics: Performance metrics dict.
    """
    st.subheader("Performance Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        st.metric("Total Trades", f"{int(metrics['total_trades'])}")
    with col2:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        st.metric("Avg Return / Trade", f"{metrics['avg_return']:+.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")


def _render_personality_profile(
    profile: dict[str, float | str],
    model_key: str,
) -> None:
    """Render agent personality radar chart and summary.

    Args:
        profile: Personality profile dict.
        model_key: ModelProvider value for colour lookup.
    """
    st.subheader("Agent Personality Profile")

    col_radar, col_summary = st.columns([3, 2])

    with col_radar:
        categories: list[str] = [
            "Contrarian", "Confidence", "Risk Appetite",
            "Decisiveness", "Consistency",
        ]
        values: list[float] = [
            float(profile["contrarian_score"]),
            float(profile["avg_confidence"]) * 100,
            float(profile["risk_appetite"]),
            float(profile["decisiveness"]),
            float(profile["consistency"]),
        ]

        agent_color: str = MODEL_COLORS.get(model_key, "#FFFFFF")
        label: str = model_key.capitalize()

        fig: go.Figure = create_radar_chart(
            categories,
            {label: values},
            "Personality Radar",
        )
        # Override fill colour
        fig.update_traces(
            fillcolor=agent_color,
            line_color=agent_color,
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_summary:
        st.markdown("#### Profile Summary")
        st.markdown(f"**Contrarian Score:** {profile['contrarian_score']}/100")
        st.markdown(f"**Avg Confidence:** {float(profile['avg_confidence']):.0%}")
        st.markdown(f"**Preferred Action:** {profile['preferred_action']}")
        st.markdown(f"**Risk Appetite:** {profile['risk_appetite']}/100")
        st.markdown(f"**Decisiveness:** {profile['decisiveness']}/100")
        st.markdown(f"**Consistency:** {profile['consistency']}/100")

        # Textual personality interpretation
        contrarian: float = float(profile["contrarian_score"])
        risk: float = float(profile["risk_appetite"])

        if contrarian > 60:
            style: str = "This agent tends to go against market consensus."
        elif contrarian < 40:
            style = "This agent follows market trends and momentum."
        else:
            style = "This agent shows balanced trend/contrarian tendencies."

        if risk > 60:
            style += " It favours aggressive position sizing."
        elif risk < 40:
            style += " It maintains conservative position sizing."
        else:
            style += " It uses moderate position sizing."

        st.info(style)


# ── Page Entry Point ──────────────────────────────────────────────


def render() -> None:
    """Render the Agent Detail page."""
    st.header("Agent Detail")

    # ---- Agent selector ----
    col_model, col_arch = st.columns(2)
    with col_model:
        selected_model_label: str = st.selectbox(
            "Model",
            options=[m.value.capitalize() for m in _MODELS],
            index=0,
            key="agent_detail_model",
        )
    with col_arch:
        selected_arch_label: str = st.selectbox(
            "Architecture",
            options=[a.value.capitalize() for a in _ARCHS],
            index=0,
            key="agent_detail_arch",
        )

    model_key: str = selected_model_label.lower()
    arch_key: str = selected_arch_label.lower()
    st.caption(
        f"Showing details for **{selected_model_label} {selected_arch_label}** | "
        f"Updated: {format_kst(datetime.now(timezone.utc))}"
    )

    st.divider()

    # ---- Load demo data ----
    decisions: list[dict[str, str | float | datetime]] = _generate_decision_timeline(
        model_key, arch_key,
    )
    metrics: dict[str, float] = _generate_performance_metrics(model_key, arch_key)
    profile: dict[str, float | str] = _generate_personality_profile(model_key, arch_key)

    # ---- Metrics row ----
    _render_metrics(metrics)

    st.divider()

    # ---- Decision timeline + Reasoning ----
    _render_decision_timeline(decisions)
    _render_reasoning_browser(decisions)

    st.divider()

    # ---- Action distribution + Confidence histogram ----
    col_pie, col_hist = st.columns(2)
    with col_pie:
        _render_action_distribution(decisions)
    with col_hist:
        _render_confidence_histogram(decisions)

    st.divider()

    # ---- Personality profile ----
    _render_personality_profile(profile, model_key)

    logger.debug(
        "agent_detail_rendered",
        model=model_key,
        architecture=arch_key,
    )
