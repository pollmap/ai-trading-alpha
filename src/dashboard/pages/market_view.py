"""Market View page — per-market equity curves, heatmaps, top movers, and insights.

Provides a tabbed interface (KRX / US / CRYPTO) with:
- Equity curves for all 9 agent combinations per market
- Symbol heatmap showing price changes
- Top movers table
- Market-specific insight panels
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from src.core.constants import MODEL_COLORS
from src.core.types import AgentArchitecture, Market, ModelProvider
from src.dashboard.components.charts import (
    CHART_LAYOUT,
    create_equity_curve,
    create_heatmap,
    format_kst,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── Agent label helpers ─────────────────────────────────────────────

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


def _agent_label(model: ModelProvider, arch: AgentArchitecture) -> str:
    """Build a human-readable label like ``DeepSeek Single``."""
    return f"{model.value.capitalize()} {arch.value.capitalize()}"


def _agent_color(model: ModelProvider) -> str:
    """Look up the dashboard colour for a model."""
    return MODEL_COLORS.get(model.value, "#FFFFFF")


# ── Demo Data Generators ───────────────────────────────────────────

_MARKET_SYMBOLS: dict[Market, list[str]] = {
    Market.KRX: [
        "005930.KS", "000660.KS", "035420.KS", "051910.KS",
        "006400.KS", "035720.KS", "068270.KS", "105560.KS",
    ],
    Market.US: [
        "AAPL", "MSFT", "GOOGL", "AMZN",
        "NVDA", "META", "TSLA", "JPM",
    ],
    Market.CRYPTO: [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
        "XRP/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT",
    ],
}

_MARKET_CURRENCY: dict[Market, str] = {
    Market.KRX: "KRW",
    Market.US: "USD",
    Market.CRYPTO: "USDT",
}


@st.cache_data(ttl=60)
def _generate_equity_data(
    market_key: str,
) -> dict[str, dict[str, list[datetime] | list[float]]]:
    """Generate demo equity-curve data for all 9 agents in a market.

    Args:
        market_key: Market enum value used as cache key.

    Returns:
        Dict mapping agent labels to date/value series.
    """
    random.seed(hash(market_key) % (2**31))
    now_utc: datetime = datetime.now(timezone.utc)
    n_points: int = 30
    dates: list[datetime] = [
        now_utc - timedelta(days=n_points - i) for i in range(n_points)
    ]

    initial: float = 100_000.0
    result: dict[str, dict[str, list[datetime] | list[float]]] = {}

    for model in _MODELS:
        for arch in _ARCHS:
            label: str = _agent_label(model, arch)
            values: list[float] = [initial]
            for _ in range(n_points - 1):
                daily_return: float = random.gauss(0.002, 0.015)
                values.append(values[-1] * (1 + daily_return))
            result[label] = {"dates": dates, "values": values}

    # Buy & Hold baseline
    bh_values: list[float] = [initial]
    for _ in range(n_points - 1):
        bh_values.append(bh_values[-1] * (1 + random.gauss(0.001, 0.012)))
    result["Buy & Hold"] = {"dates": dates, "values": bh_values}

    return result


@st.cache_data(ttl=60)
def _generate_symbol_heatmap_data(
    market_key: str,
) -> tuple[list[list[float]], list[str], list[str]]:
    """Generate demo heatmap data for a market's symbols.

    Args:
        market_key: Market enum value.

    Returns:
        Tuple of (z_data, x_labels, y_labels).
    """
    random.seed(hash(market_key + "_heatmap") % (2**31))
    market: Market = Market(market_key)
    symbols: list[str] = _MARKET_SYMBOLS[market]

    categories: list[str] = ["1D Change", "5D Change", "Volume vs Avg"]
    z_data: list[list[float]] = []
    for _ in categories:
        row: list[float] = [round(random.uniform(-5.0, 5.0), 2) for _ in symbols]
        z_data.append(row)

    return z_data, symbols, categories


@st.cache_data(ttl=60)
def _generate_top_movers(market_key: str) -> pd.DataFrame:
    """Generate a demo top-movers table.

    Args:
        market_key: Market enum value.

    Returns:
        DataFrame with columns: Symbol, Price, Change%, Volume.
    """
    random.seed(hash(market_key + "_movers") % (2**31))
    market: Market = Market(market_key)
    symbols: list[str] = _MARKET_SYMBOLS[market]
    currency: str = _MARKET_CURRENCY[market]

    rows: list[dict[str, str | float]] = []
    for sym in symbols:
        if currency == "KRW":
            price: float = round(random.uniform(30000, 500000), 0)
        elif currency == "USDT":
            price = round(random.uniform(0.5, 95000), 2)
        else:
            price = round(random.uniform(50, 800), 2)

        change: float = round(random.uniform(-6.0, 6.0), 2)
        volume: float = round(random.uniform(1e6, 5e8), 0)
        rows.append({
            "Symbol": sym,
            "Price": price,
            "Change (%)": change,
            "Volume": volume,
        })

    df: pd.DataFrame = pd.DataFrame(rows)
    df = df.sort_values("Change (%)", ascending=False).reset_index(drop=True)
    return df


# ── Market-Specific Insight Panels ─────────────────────────────────


def _render_krx_insights() -> None:
    """KRX-specific panel: foreign/institutional flow demo data."""
    st.subheader("Foreign & Institutional Flows")
    random.seed(42)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Foreign Net Buy (today)",
            value=f"{random.randint(-500, 500):,} B KRW",
            delta=f"{random.randint(-100, 100):,} B vs yesterday",
        )
        st.metric(
            label="Foreign Ownership Ratio",
            value=f"{random.uniform(30, 40):.1f}%",
        )
    with col2:
        st.metric(
            label="Institutional Net Buy (today)",
            value=f"{random.randint(-300, 300):,} B KRW",
            delta=f"{random.randint(-80, 80):,} B vs yesterday",
        )
        st.metric(
            label="KOSPI 200 Futures Basis",
            value=f"{random.uniform(-1.5, 1.5):.2f}%",
        )

    # Flow trend bar chart
    days_labels: list[str] = [f"Day {i}" for i in range(1, 8)]
    foreign_flow: list[float] = [random.randint(-400, 400) for _ in days_labels]
    inst_flow: list[float] = [random.randint(-300, 300) for _ in days_labels]

    fig: go.Figure = go.Figure()
    fig.add_trace(go.Bar(
        x=days_labels, y=foreign_flow, name="Foreign",
        marker_color="#4285F4",
    ))
    fig.add_trace(go.Bar(
        x=days_labels, y=inst_flow, name="Institutional",
        marker_color="#D97706",
    ))
    fig.update_layout(
        title={"text": "7-Day Flow Trend (Billion KRW)", "x": 0.02, "xanchor": "left"},
        barmode="group",
        height=350,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_us_insights() -> None:
    """US-specific panel: sector performance demo data."""
    st.subheader("S&P 500 Sector Performance")
    random.seed(43)

    sectors: list[str] = [
        "Technology", "Healthcare", "Financials", "Consumer Disc.",
        "Industrials", "Energy", "Comm. Services", "Utilities",
        "Materials", "Real Estate", "Consumer Staples",
    ]
    performances: list[float] = [round(random.uniform(-3.0, 4.0), 2) for _ in sectors]

    bar_colors: list[str] = [
        "#22C55E" if p >= 0 else "#EF4444" for p in performances
    ]

    fig: go.Figure = go.Figure(
        data=go.Bar(
            x=performances,
            y=sectors,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p:+.2f}%" for p in performances],
            textposition="outside",
        )
    )
    fig.update_layout(
        title={"text": "Sector Performance (1D)", "x": 0.02, "xanchor": "left"},
        xaxis_title="Change (%)",
        height=450,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("VIX", f"{random.uniform(12, 28):.1f}", f"{random.uniform(-2, 2):+.1f}")
    with col2:
        st.metric("10Y Treasury", f"{random.uniform(3.8, 4.8):.2f}%", f"{random.uniform(-0.1, 0.1):+.2f}%")
    with col3:
        st.metric("USD Index", f"{random.uniform(100, 108):.1f}", f"{random.uniform(-0.5, 0.5):+.1f}")


def _render_crypto_insights() -> None:
    """Crypto-specific panel: 24h volume and dominance demo data."""
    st.subheader("Crypto Market Overview")
    random.seed(44)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Market Cap",
            f"${random.uniform(2.0, 3.5):.2f}T",
            f"{random.uniform(-3, 3):+.1f}%",
        )
    with col2:
        st.metric(
            "24h Volume",
            f"${random.uniform(60, 150):.0f}B",
            f"{random.uniform(-10, 15):+.1f}%",
        )
    with col3:
        st.metric(
            "BTC Dominance",
            f"{random.uniform(48, 58):.1f}%",
            f"{random.uniform(-1, 1):+.1f}%",
        )
    with col4:
        st.metric(
            "Fear & Greed",
            f"{random.randint(20, 80)}",
            f"{random.randint(-10, 10):+d} vs yesterday",
        )

    # 24h volume bar chart per token
    tokens: list[str] = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]
    volumes: list[float] = [round(random.uniform(1, 40), 1) for _ in tokens]

    fig: go.Figure = go.Figure(
        data=go.Bar(
            x=tokens,
            y=volumes,
            marker_color=["#F7931A", "#627EEA", "#9945FF", "#F0B90B",
                          "#23292F", "#0033AD", "#C2A633", "#E84142"],
            text=[f"${v:.1f}B" for v in volumes],
            textposition="outside",
        )
    )
    fig.update_layout(
        title={"text": "24h Trading Volume (Billion USD)", "x": 0.02, "xanchor": "left"},
        yaxis_title="Volume ($B)",
        height=350,
        **CHART_LAYOUT,  # type: ignore[arg-type]
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Colour mapping for equity curves ───────────────────────────────


def _build_color_map() -> dict[str, str]:
    """Map each agent label to its dashboard colour."""
    color_map: dict[str, str] = {}
    for model in _MODELS:
        for arch in _ARCHS:
            label: str = _agent_label(model, arch)
            color_map[label] = _agent_color(model)
    color_map["Buy & Hold"] = MODEL_COLORS["buy_hold"]
    return color_map


# ── Single Market Renderer ─────────────────────────────────────────

_INSIGHT_RENDERERS: dict[Market, type[None]] = {}  # populated below


def _render_market_tab(market: Market) -> None:
    """Render a complete market tab with equity curves, heatmap, movers, and insights.

    Args:
        market: Which market to render.
    """
    market_key: str = market.value

    # Timestamp header
    now_utc: datetime = datetime.now(timezone.utc)
    st.caption(f"Last updated: {format_kst(now_utc)}")

    # ---- Equity curves ----
    st.subheader(f"{market_key} Equity Curves (9 Agents)")
    equity_data: dict[str, dict[str, list[datetime] | list[float]]] = (
        _generate_equity_data(market_key)
    )
    color_map: dict[str, str] = _build_color_map()
    fig_equity: go.Figure = create_equity_curve(
        equity_data,
        color_map,
        f"{market_key} — Agent Performance",
    )
    st.plotly_chart(fig_equity, use_container_width=True)

    # ---- Symbol heatmap ----
    st.subheader(f"{market_key} Symbol Heatmap")
    z_data, x_labels, y_labels = _generate_symbol_heatmap_data(market_key)
    fig_heatmap: go.Figure = create_heatmap(z_data, x_labels, y_labels, f"{market_key} Price Changes")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # ---- Top movers ----
    st.subheader(f"{market_key} Top Movers")
    movers_df: pd.DataFrame = _generate_top_movers(market_key)
    st.dataframe(
        movers_df.style.applymap(  # type: ignore[arg-type]
            lambda v: "color: #22C55E" if isinstance(v, (int, float)) and v > 0
            else ("color: #EF4444" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["Change (%)"],
        ),
        use_container_width=True,
        height=340,
    )

    # ---- Market-specific insights ----
    st.divider()
    if market == Market.KRX:
        _render_krx_insights()
    elif market == Market.US:
        _render_us_insights()
    elif market == Market.CRYPTO:
        _render_crypto_insights()


# ── Page Entry Point ───────────────────────────────────────────────


def render() -> None:
    """Render the Market View page with KRX / US / CRYPTO tabs."""
    st.header("Market View")

    tab_krx, tab_us, tab_crypto = st.tabs(["KRX", "US", "CRYPTO"])

    with tab_krx:
        _render_market_tab(Market.KRX)
    with tab_us:
        _render_market_tab(Market.US)
    with tab_crypto:
        _render_market_tab(Market.CRYPTO)

    logger.debug("market_view_rendered")
