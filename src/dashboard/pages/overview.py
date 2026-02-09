"""Overview page — 9-portfolio equity curves, metric cards, and summary table.

Displays all agent combinations (4 LLMs x 2 architectures + Buy&Hold) with
Plotly charts.  Uses real data from ResultsStore when available, falls back
to demo data otherwise.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
from src.core.types import AgentArchitecture, Market, ModelProvider
from src.dashboard.data_loader import (
    has_real_data,
    load_equity_curves_df,
    load_signals_df,
    load_status,
    build_portfolio_label,
    get_latest_portfolios,
)

log: structlog.stdlib.BoundLogger = structlog.get_logger("dashboard.overview")

# ── Timezone ────────────────────────────────────────────────────
KST: timezone = timezone(timedelta(hours=9))

# ── Portfolio label helpers ─────────────────────────────────────
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

INITIAL_CAPITAL: float = 10_000_000.0  # 10M (KRW-scale demo)
NUM_DAYS: int = 30


# ── Demo data generation (fallback) ──────────────────────────────


def _label(model: str, arch: str) -> str:
    """Create a human-readable label for a model/architecture pair."""
    return f"{model.capitalize()} ({arch.capitalize()})"


@st.cache_data(ttl=60)
def _generate_demo_equity_curves() -> pd.DataFrame:
    """Generate synthetic equity curves for 9 portfolios over *NUM_DAYS* days."""
    rng: np.random.Generator = np.random.default_rng(seed=42)
    dates: list[datetime] = [
        datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        - timedelta(days=NUM_DAYS - i)
        for i in range(NUM_DAYS)
    ]

    rows: list[dict[str, object]] = []

    for model, arch in AGENTS:
        drift: float = rng.uniform(-0.001, 0.003)
        vol: float = rng.uniform(0.008, 0.025)
        daily_returns: np.ndarray = rng.normal(drift, vol, size=NUM_DAYS)  # type: ignore[assignment]
        cum_value: float = INITIAL_CAPITAL
        for i, dt in enumerate(dates):
            if i > 0:
                cum_value *= 1.0 + daily_returns[i]
            rows.append(
                {"date": dt, "portfolio": _label(model, arch), "value": cum_value}
            )

    bh_drift: float = 0.0005
    bh_vol: float = 0.012
    bh_returns: np.ndarray = rng.normal(bh_drift, bh_vol, size=NUM_DAYS)  # type: ignore[assignment]
    cum_bh: float = INITIAL_CAPITAL
    for i, dt in enumerate(dates):
        if i > 0:
            cum_bh *= 1.0 + bh_returns[i]
        rows.append({"date": dt, "portfolio": "Buy & Hold", "value": cum_bh})

    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def _generate_demo_metrics() -> pd.DataFrame:
    """Generate synthetic performance metrics for each portfolio."""
    rng: np.random.Generator = np.random.default_rng(seed=42)
    rows: list[dict[str, object]] = []

    for model, arch in AGENTS:
        total_ret: float = rng.uniform(-5.0, 15.0)
        sharpe: float = rng.uniform(-0.5, 2.5)
        mdd: float = rng.uniform(-20.0, -2.0)
        win_rate: float = rng.uniform(40.0, 65.0)
        latency: float = rng.uniform(200.0, 2000.0)
        rows.append(
            {
                "portfolio": _label(model, arch),
                "total_return_pct": round(total_ret, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown_pct": round(mdd, 2),
                "win_rate_pct": round(win_rate, 1),
                "avg_latency_ms": round(latency, 0),
                "model": model,
                "architecture": arch,
            }
        )

    rows.append(
        {
            "portfolio": "Buy & Hold",
            "total_return_pct": round(rng.uniform(-2.0, 8.0), 2),
            "sharpe_ratio": round(rng.uniform(0.0, 1.5), 2),
            "max_drawdown_pct": round(rng.uniform(-15.0, -3.0), 2),
            "win_rate_pct": 0.0,
            "avg_latency_ms": 0.0,
            "model": "buy_hold",
            "architecture": "",
        }
    )

    return pd.DataFrame(rows)


# ── Real data processing ─────────────────────────────────────────


@st.cache_data(ttl=30)
def _load_real_equity_curves() -> pd.DataFrame | None:
    """Load real equity curve data from results store."""
    df = load_equity_curves_df()
    if df is None or df.empty:
        return None
    return df


@st.cache_data(ttl=30)
def _build_real_metrics(equity_json: str) -> pd.DataFrame:
    """Build performance metrics from real equity curve data.

    Takes JSON string to allow st.cache_data hashing.
    """
    equity_df = pd.read_json(equity_json)
    rows: list[dict[str, object]] = []

    for (model, arch, market), group in equity_df.groupby(["model", "architecture", "market"]):
        group = group.sort_values("cycle")
        if len(group) < 2:
            continue

        values = group["total_value"].values
        initial = group["initial_capital"].values[0]

        total_ret = (values[-1] / initial - 1) * 100 if initial > 0 else 0.0

        returns = np.diff(values) / values[:-1]
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0

        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax
        mdd = drawdowns.min() * 100

        label = build_portfolio_label(str(model), str(arch))
        rows.append({
            "portfolio": label,
            "total_return_pct": round(float(total_ret), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(mdd), 2),
            "win_rate_pct": 0.0,
            "avg_latency_ms": 0.0,
            "model": model,
            "architecture": arch,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ── Chart builders ──────────────────────────────────────────────


def _build_equity_chart(df: pd.DataFrame) -> go.Figure:
    """Build a Plotly line chart with 9 equity curves (demo mode)."""
    fig: go.Figure = go.Figure()

    for model, arch in AGENTS:
        label: str = _label(model, arch)
        color: str = MODEL_COLORS.get(model, "#888888")
        dash: str = "solid" if arch == ARCH_SINGLE else "dash"
        subset: pd.DataFrame = df[df["portfolio"] == label]

        display_dates: list[str] = [
            dt.replace(tzinfo=timezone.utc).astimezone(KST).strftime("%m/%d %H:%M")
            if isinstance(dt, datetime)
            else str(dt)
            for dt in subset["date"]
        ]

        fig.add_trace(
            go.Scatter(
                x=display_dates,
                y=subset["value"],
                mode="lines",
                name=label,
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Date: %{x}<br>"
                    "Value: %{y:,.0f}<extra></extra>"
                ),
            )
        )

    bh_subset: pd.DataFrame = df[df["portfolio"] == "Buy & Hold"]
    bh_dates: list[str] = [
        dt.replace(tzinfo=timezone.utc).astimezone(KST).strftime("%m/%d %H:%M")
        if isinstance(dt, datetime)
        else str(dt)
        for dt in bh_subset["date"]
    ]
    fig.add_trace(
        go.Scatter(
            x=bh_dates,
            y=bh_subset["value"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color=MODEL_COLORS["buy_hold"], width=2, dash="dot"),
            hovertemplate=(
                "<b>Buy & Hold</b><br>"
                "Date: %{x}<br>"
                "Value: %{y:,.0f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Portfolio Equity Curves (All Agents)",
        xaxis_title="Date (KST)",
        yaxis_title="Portfolio Value",
        yaxis_tickformat=",",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
        template="plotly_white",
        height=550,
        margin=dict(b=120),
    )

    fig.add_hline(
        y=INITIAL_CAPITAL,
        line_dash="dot",
        line_color="rgba(0,0,0,0.2)",
        annotation_text="Initial Capital",
        annotation_position="top left",
    )

    return fig


def _build_real_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    """Build equity chart from real results data."""
    fig = go.Figure()

    for (model, arch), group in equity_df.groupby(["model", "architecture"]):
        group = group.sort_values("cycle")
        label = build_portfolio_label(str(model), str(arch))
        color = MODEL_COLORS.get(str(model), "#888888")
        dash = "solid" if str(arch) == ARCH_SINGLE else "dash"

        timestamps = pd.to_datetime(group["timestamp"])
        display_dates = [
            ts.tz_localize("UTC").astimezone(KST).strftime("%m/%d %H:%M")
            if ts.tzinfo is None
            else ts.astimezone(KST).strftime("%m/%d %H:%M")
            for ts in timestamps
        ]

        fig.add_trace(go.Scatter(
            x=display_dates,
            y=group["total_value"],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2, dash=dash),
            marker=dict(size=4),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Cycle: %{customdata}<br>"
                "Value: %{y:,.0f}<extra></extra>"
            ),
            customdata=group["cycle"],
        ))

    initial = equity_df["initial_capital"].iloc[0] if not equity_df.empty else 100_000

    fig.update_layout(
        title="Portfolio Equity Curves (Live Data)",
        xaxis_title="Time (KST)",
        yaxis_title="Portfolio Value",
        yaxis_tickformat=",",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5,
            font=dict(size=10),
        ),
        hovermode="x unified",
        template="plotly_white",
        height=550,
        margin=dict(b=120),
    )

    fig.add_hline(
        y=initial,
        line_dash="dot",
        line_color="rgba(0,0,0,0.2)",
        annotation_text="Initial Capital",
        annotation_position="top left",
    )

    return fig


# ── Metric cards ────────────────────────────────────────────────


def _render_metric_cards(metrics_df: pd.DataFrame) -> None:
    """Render metric summary cards for each model (best of single/multi)."""
    models: list[str] = [MODEL_DEEPSEEK, MODEL_GEMINI, MODEL_CLAUDE, MODEL_GPT, "buy_hold"]
    display_names: dict[str, str] = {
        MODEL_DEEPSEEK: "DeepSeek",
        MODEL_GEMINI: "Gemini",
        MODEL_CLAUDE: "Claude",
        MODEL_GPT: "GPT",
        "buy_hold": "Buy & Hold",
    }

    cols: list[object] = st.columns(len(models))

    for idx, model in enumerate(models):
        model_rows: pd.DataFrame = metrics_df[metrics_df["model"] == model]
        if model_rows.empty:
            continue

        best_row: pd.Series = model_rows.loc[model_rows["total_return_pct"].idxmax()]  # type: ignore[assignment]
        color: str = MODEL_COLORS.get(model, "#888888")

        with cols[idx]:  # type: ignore[index]
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 12px;'>"
                f"<h4 style='margin-bottom:0'>{display_names.get(model, model)}</h4>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.metric(
                label="Total Return",
                value=f"{best_row['total_return_pct']:+.2f}%",
            )
            st.metric(
                label="Sharpe Ratio",
                value=f"{best_row['sharpe_ratio']:.2f}",
            )
            st.metric(
                label="Max Drawdown",
                value=f"{best_row['max_drawdown_pct']:.2f}%",
            )


# ── Summary table ───────────────────────────────────────────────


def _render_summary_table(metrics_df: pd.DataFrame) -> None:
    """Render a sortable summary table comparing all 9 portfolios."""
    columns_to_show = ["portfolio", "total_return_pct", "sharpe_ratio", "max_drawdown_pct"]
    for col in ["win_rate_pct", "avg_latency_ms"]:
        if col in metrics_df.columns:
            columns_to_show.append(col)

    display_df: pd.DataFrame = metrics_df[
        [c for c in columns_to_show if c in metrics_df.columns]
    ].copy()

    rename_map = {
        "portfolio": "Portfolio",
        "total_return_pct": "Return (%)",
        "sharpe_ratio": "Sharpe",
        "max_drawdown_pct": "MDD (%)",
        "win_rate_pct": "Win Rate (%)",
        "avg_latency_ms": "Avg Latency (ms)",
    }
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

    sort_col = "Return (%)" if "Return (%)" in display_df.columns else display_df.columns[1]
    display_df = display_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Page entry point ────────────────────────────────────────────


def render() -> None:
    """Render the Overview page."""
    st.header("Overview")

    use_real = has_real_data()

    if use_real:
        st.success("LIVE DATA — Connected to benchmark results")
        status = load_status()
        if status:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cycles", status.get("cycle_count", 0))
            with col2:
                st.metric("Markets", ", ".join(status.get("markets", [])))
            with col3:
                st.metric("Agents", len(status.get("registered_agents", [])))
            with col4:
                running = status.get("running", False)
                st.metric("Status", "Running" if running else "Stopped")
    else:
        st.info(
            "DEMO MODE — No benchmark results found. "
            "Run `make run` to start the benchmark and see live data."
        )

    st.caption(
        "9 portfolios: 4 LLMs (DeepSeek, Gemini, Claude, GPT) x 2 architectures "
        "(Single, Multi) + Buy & Hold baseline"
    )

    log.info("rendering_overview_page", live_data=use_real)

    market_choice: str = st.selectbox(
        "Market",
        options=[m.value for m in Market],
        index=0,
        key="overview_market",
    )

    if use_real:
        equity_df = _load_real_equity_curves()
        if equity_df is not None and not equity_df.empty:
            market_df = equity_df[equity_df["market"] == market_choice]
            if not market_df.empty:
                st.subheader("Equity Curves")
                st.caption("Solid = Single agent | Dashed = Multi agent")
                fig = _build_real_equity_chart(market_df)
                st.plotly_chart(fig, use_container_width=True)

                metrics_df = _build_real_metrics(market_df.to_json())
                if not metrics_df.empty:
                    st.subheader("Performance Highlights")
                    _render_metric_cards(metrics_df)
                    st.subheader("Agent Comparison Table")
                    _render_summary_table(metrics_df)
            else:
                st.warning(f"No data for market: {market_choice}")
        else:
            st.warning("Results store is empty. Run a benchmark cycle first.")
    else:
        equity_df = _generate_demo_equity_curves()
        metrics_df = _generate_demo_metrics()

        st.subheader("Equity Curves")
        st.caption("Solid = Single agent | Dashed = Multi agent | Dotted = Buy & Hold")
        fig = _build_equity_chart(equity_df)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Highlights")
        _render_metric_cards(metrics_df)

        st.subheader("Agent Comparison Table")
        _render_summary_table(metrics_df)
