"""System-wide shared types — the single source of truth for all data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from textwrap import dedent


# ── Enums ────────────────────────────────────────────────────────

class Market(str, Enum):
    KRX = "KRX"
    US = "US"
    CRYPTO = "CRYPTO"
    JPX = "JPX"               # Tokyo Stock Exchange (Japan)
    SSE = "SSE"               # Shanghai Stock Exchange (China)
    HKEX = "HKEX"             # Hong Kong Exchange
    EURONEXT = "EURONEXT"     # Pan-European (Paris, Amsterdam, Brussels, Lisbon)
    LSE = "LSE"               # London Stock Exchange
    BOND = "BOND"             # Global bond market
    COMMODITIES = "COMMODITIES"  # Commodities (gold, oil, etc.)


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ModelProvider(str, Enum):
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    CLAUDE = "claude"
    GPT = "gpt"


class AgentArchitecture(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


# ── Market Data Types ────────────────────────────────────────────

@dataclass
class SymbolData:
    """OHLCV + fundamentals for a single symbol."""

    symbol: str
    market: Market
    open: float
    high: float
    low: float
    close: float
    volume: float
    currency: str  # "KRW", "USD", "USDT"
    per: float | None = None
    pbr: float | None = None
    market_cap: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.close < 0:
            msg = f"close price cannot be negative: {self.close}"
            raise ValueError(msg)
        if self.volume < 0:
            msg = f"volume cannot be negative: {self.volume}"
            raise ValueError(msg)


@dataclass
class MacroData:
    """Macroeconomic indicators across markets."""

    kr_base_rate: float | None = None
    us_fed_rate: float | None = None
    usdkrw: float | None = None
    vix: float | None = None
    kr_cpi_yoy: float | None = None
    us_cpi_yoy: float | None = None
    fear_greed_index: float | None = None
    # Global macro fields (v4 expansion)
    jp_base_rate: float | None = None
    cn_base_rate: float | None = None
    ecb_rate: float | None = None
    boe_rate: float | None = None
    usdjpy: float | None = None
    usdcny: float | None = None
    eurusd: float | None = None
    gbpusd: float | None = None
    usdhkd: float | None = None
    gold_price: float | None = None
    oil_wti_price: float | None = None
    us_10y_yield: float | None = None
    jp_10y_yield: float | None = None
    de_10y_yield: float | None = None
    cn_10y_yield: float | None = None


@dataclass
class NewsItem:
    """Single news/sentiment item."""

    timestamp: datetime
    title: str
    summary: str
    source: str
    relevance_score: float = 0.0  # 0~1
    sentiment: float = 0.0        # -1~+1

    def __post_init__(self) -> None:
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))
        self.sentiment = max(-1.0, min(1.0, self.sentiment))


@dataclass
class MarketSnapshot:
    """Immutable standardized payload: MIS -> AEP.

    Once created, this object must NOT be modified.
    """

    snapshot_id: str
    timestamp: datetime
    market: Market
    symbols: dict[str, SymbolData]
    macro: MacroData
    news: list[NewsItem] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            msg = "MarketSnapshot timestamp must be timezone-aware (UTC)"
            raise ValueError(msg)

    def to_prompt_summary(self) -> str:
        """Token-efficient text summary for LLM prompts."""
        ts_str = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"[Market: {self.market.value} | Time: {ts_str}]"]
        lines.append("")

        # Symbols summary
        lines.append("=== Symbols ===")
        for sym, sd in self.symbols.items():
            chg = ((sd.close - sd.open) / sd.open * 100) if sd.open else 0.0
            line = f"{sym}: O={sd.open:.2f} H={sd.high:.2f} L={sd.low:.2f} C={sd.close:.2f} V={sd.volume:.0f} ({chg:+.2f}%)"
            if sd.per is not None:
                line += f" PER={sd.per:.1f}"
            if sd.pbr is not None:
                line += f" PBR={sd.pbr:.2f}"
            lines.append(line)

        # Macro summary
        lines.append("")
        lines.append("=== Macro ===")
        macro = self.macro
        macro_items: list[str] = []
        if macro.kr_base_rate is not None:
            macro_items.append(f"KR Base Rate={macro.kr_base_rate:.2f}%")
        if macro.us_fed_rate is not None:
            macro_items.append(f"US Fed Rate={macro.us_fed_rate:.2f}%")
        if macro.jp_base_rate is not None:
            macro_items.append(f"JP Base Rate={macro.jp_base_rate:.2f}%")
        if macro.cn_base_rate is not None:
            macro_items.append(f"CN Base Rate={macro.cn_base_rate:.2f}%")
        if macro.ecb_rate is not None:
            macro_items.append(f"ECB Rate={macro.ecb_rate:.2f}%")
        if macro.boe_rate is not None:
            macro_items.append(f"BoE Rate={macro.boe_rate:.2f}%")
        if macro.usdkrw is not None:
            macro_items.append(f"USD/KRW={macro.usdkrw:.1f}")
        if macro.usdjpy is not None:
            macro_items.append(f"USD/JPY={macro.usdjpy:.2f}")
        if macro.usdcny is not None:
            macro_items.append(f"USD/CNY={macro.usdcny:.4f}")
        if macro.eurusd is not None:
            macro_items.append(f"EUR/USD={macro.eurusd:.4f}")
        if macro.gbpusd is not None:
            macro_items.append(f"GBP/USD={macro.gbpusd:.4f}")
        if macro.usdhkd is not None:
            macro_items.append(f"USD/HKD={macro.usdhkd:.4f}")
        if macro.vix is not None:
            macro_items.append(f"VIX={macro.vix:.1f}")
        if macro.fear_greed_index is not None:
            macro_items.append(f"Fear&Greed={macro.fear_greed_index:.0f}")
        if macro.gold_price is not None:
            macro_items.append(f"Gold={macro.gold_price:.2f}")
        if macro.oil_wti_price is not None:
            macro_items.append(f"WTI Oil={macro.oil_wti_price:.2f}")
        if macro.us_10y_yield is not None:
            macro_items.append(f"US 10Y={macro.us_10y_yield:.3f}%")
        if macro.jp_10y_yield is not None:
            macro_items.append(f"JP 10Y={macro.jp_10y_yield:.3f}%")
        if macro.de_10y_yield is not None:
            macro_items.append(f"DE 10Y={macro.de_10y_yield:.3f}%")
        if macro.cn_10y_yield is not None:
            macro_items.append(f"CN 10Y={macro.cn_10y_yield:.3f}%")
        lines.append(", ".join(macro_items) if macro_items else "No macro data")

        # News summary (top 5)
        if self.news:
            lines.append("")
            lines.append("=== Recent News (top 5) ===")
            for item in sorted(self.news, key=lambda n: n.relevance_score, reverse=True)[:5]:
                lines.append(f"- [{item.source}] {item.title}")

        return "\n".join(lines)


# ── Trading Types ────────────────────────────────────────────────

@dataclass
class TradingSignal:
    """Trading signal returned by each agent."""

    signal_id: str
    snapshot_id: str
    timestamp: datetime
    symbol: str
    market: Market
    action: Action
    weight: float         # 0~1 (portfolio weight)
    confidence: float     # 0~1
    reasoning: str        # decision rationale (must not be empty)
    model: ModelProvider
    architecture: AgentArchitecture
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weight = max(0.0, min(1.0, self.weight))
        self.confidence = max(0.0, min(1.0, self.confidence))
        if not self.reasoning.strip():
            msg = "TradingSignal reasoning must not be empty"
            raise ValueError(msg)


# ── Portfolio Types ──────────────────────────────────────────────

@dataclass
class Position:
    """Single position in a portfolio."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        if self.quantity > 0 and self.avg_entry_price > 0:
            self.unrealized_pnl = (self.current_price - self.avg_entry_price) * self.quantity


@dataclass
class PortfolioState:
    """Independent portfolio per agent combination."""

    portfolio_id: str
    model: ModelProvider
    architecture: AgentArchitecture
    market: Market
    cash: float
    positions: dict[str, Position]
    initial_capital: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def total_value(self) -> float:
        """Total portfolio value = cash + sum of all position values."""
        position_value = sum(
            pos.current_price * pos.quantity for pos in self.positions.values()
        )
        return self.cash + position_value

    @property
    def cash_ratio(self) -> float:
        """Cash as a fraction of total portfolio value."""
        total = self.total_value
        if total <= 0:
            return 1.0
        return self.cash / total

    def to_prompt_summary(self) -> str:
        """Token-efficient portfolio summary for LLM prompts."""
        total = self.total_value
        pnl_pct = ((total / self.initial_capital) - 1.0) * 100 if self.initial_capital else 0.0

        lines = [
            f"[Portfolio: {self.model.value}/{self.architecture.value} | {self.market.value}]",
            f"Total Value: {total:,.0f} ({pnl_pct:+.2f}% from initial)",
            f"Cash: {self.cash:,.0f} ({self.cash_ratio:.1%})",
        ]

        if self.positions:
            lines.append("Positions:")
            for sym, pos in self.positions.items():
                pos_pnl_pct = (
                    (pos.current_price / pos.avg_entry_price - 1) * 100
                    if pos.avg_entry_price
                    else 0.0
                )
                lines.append(
                    f"  {sym}: qty={pos.quantity:.4f} avg={pos.avg_entry_price:.2f} "
                    f"cur={pos.current_price:.2f} ({pos_pnl_pct:+.2f}%)"
                )
        else:
            lines.append("Positions: None")

        return "\n".join(lines)
