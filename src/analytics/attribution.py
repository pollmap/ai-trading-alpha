"""Decision attribution — trace each trading signal to its data sources.

For each trade, records which data inputs (price data, macro, news, indicators,
social sentiment) influenced the decision and by how much, enabling:
- Understanding which data sources drive returns
- Identifying useless or harmful data inputs
- Debugging unexpected agent behaviour
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.core.logging import get_logger
from src.core.types import TradingSignal, Action

log = get_logger(__name__)


class DataSource:
    """Constants for data source identification."""
    PRICE_ACTION = "price_action"
    TECHNICAL_INDICATORS = "technical_indicators"
    MACRO_DATA = "macro_data"
    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_SENTIMENT = "social_sentiment"
    ON_CHAIN = "on_chain"
    PORTFOLIO_STATE = "portfolio_state"
    SELF_REFLECTION = "self_reflection"
    REGIME_CONTEXT = "regime_context"


@dataclass
class AttributionWeight:
    """Weight of a single data source's contribution to a decision."""
    source: str
    weight: float  # 0-1, how much this source influenced the decision
    evidence: str  # Brief description of what the source contributed
    sentiment_direction: float = 0.0  # -1 (bearish) to +1 (bullish)


@dataclass
class DecisionAttribution:
    """Full attribution record for a single trading signal."""
    signal_id: str
    symbol: str
    action: Action
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributions: list[AttributionWeight] = field(default_factory=list)
    total_bullish_weight: float = 0.0
    total_bearish_weight: float = 0.0
    dominant_source: str = ""
    reasoning_summary: str = ""

    def __post_init__(self) -> None:
        if self.attributions:
            self._recalculate()

    def _recalculate(self) -> None:
        self.total_bullish_weight = sum(
            a.weight for a in self.attributions if a.sentiment_direction > 0
        )
        self.total_bearish_weight = sum(
            a.weight for a in self.attributions if a.sentiment_direction < 0
        )
        if self.attributions:
            self.dominant_source = max(
                self.attributions, key=lambda a: a.weight
            ).source


@dataclass
class AttributionSummary:
    """Aggregated attribution across multiple decisions."""
    total_decisions: int = 0
    source_win_rates: dict[str, float] = field(default_factory=dict)
    source_avg_weights: dict[str, float] = field(default_factory=dict)
    source_pnl_contribution: dict[str, float] = field(default_factory=dict)


class AttributionTracker:
    """Track and analyze decision attributions over time.

    Parses LLM reasoning to identify which data sources influenced
    each decision, then correlates with outcomes.
    """

    # Keywords that indicate a data source was referenced
    _SOURCE_KEYWORDS: dict[str, list[str]] = {
        DataSource.PRICE_ACTION: [
            "price", "close", "open", "high", "low", "candle",
            "support", "resistance", "breakout", "gap",
        ],
        DataSource.TECHNICAL_INDICATORS: [
            "rsi", "macd", "sma", "ema", "bollinger", "atr",
            "indicator", "oversold", "overbought", "crossover",
            "momentum", "moving average",
        ],
        DataSource.MACRO_DATA: [
            "fed", "rate", "cpi", "gdp", "inflation", "macro",
            "interest", "monetary", "fiscal", "unemployment",
            "vix", "yield", "treasury",
        ],
        DataSource.NEWS_SENTIMENT: [
            "news", "headline", "report", "announced", "earnings",
            "guidance", "forecast", "analyst", "upgrade", "downgrade",
        ],
        DataSource.SOCIAL_SENTIMENT: [
            "social", "reddit", "twitter", "sentiment", "buzz",
            "trending", "mentions", "retail", "wsb",
        ],
        DataSource.ON_CHAIN: [
            "whale", "on-chain", "funding rate", "exchange flow",
            "outflow", "inflow", "defi", "tvl",
        ],
        DataSource.PORTFOLIO_STATE: [
            "portfolio", "position", "cash", "exposure",
            "weight", "allocation", "drawdown",
        ],
        DataSource.SELF_REFLECTION: [
            "reflection", "past performance", "win rate",
            "losing streak", "previous trades", "calibration",
        ],
        DataSource.REGIME_CONTEXT: [
            "regime", "bull market", "bear market", "sideways",
            "volatility regime", "market phase", "cycle",
        ],
    }

    def __init__(self) -> None:
        self._history: list[tuple[DecisionAttribution, float]] = []  # (attribution, pnl)

    def attribute(
        self, signal: TradingSignal, raw_reasoning: str = ""
    ) -> DecisionAttribution:
        """Parse reasoning text and create attribution record.

        Args:
            signal: The trading signal to attribute.
            raw_reasoning: Full LLM reasoning text (defaults to signal.reasoning).

        Returns:
            DecisionAttribution with identified data source weights.
        """
        text = raw_reasoning or signal.reasoning
        text_lower = text.lower()

        attributions: list[AttributionWeight] = []
        total_mentions = 0

        for source, keywords in self._SOURCE_KEYWORDS.items():
            mention_count = sum(1 for kw in keywords if kw in text_lower)
            if mention_count > 0:
                total_mentions += mention_count
                # Determine sentiment direction from context
                direction = self._infer_direction(text_lower, keywords, signal.action)
                attributions.append(AttributionWeight(
                    source=source,
                    weight=float(mention_count),  # Will normalise below
                    evidence=f"{mention_count} keyword matches",
                    sentiment_direction=direction,
                ))

        # Normalise weights to sum to 1
        if total_mentions > 0:
            for attr in attributions:
                attr.weight = attr.weight / total_mentions

        record = DecisionAttribution(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            action=signal.action,
            attributions=attributions,
            reasoning_summary=text[:200] if text else "",
        )
        record._recalculate()

        log.info(
            "decision_attributed",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            source_count=len(attributions),
            dominant=record.dominant_source,
        )

        return record

    def record_outcome(
        self, attribution: DecisionAttribution, realized_pnl: float
    ) -> None:
        """Record the outcome of an attributed decision."""
        self._history.append((attribution, realized_pnl))
        if len(self._history) > 5000:
            self._history = self._history[-2500:]

    def summarize(self) -> AttributionSummary:
        """Generate aggregated attribution analysis."""
        if not self._history:
            return AttributionSummary()

        source_wins: dict[str, int] = {}
        source_total: dict[str, int] = {}
        source_weights: dict[str, list[float]] = {}
        source_pnl: dict[str, float] = {}

        for attr, pnl in self._history:
            for a in attr.attributions:
                src = a.source
                source_total[src] = source_total.get(src, 0) + 1
                if pnl > 0:
                    source_wins[src] = source_wins.get(src, 0) + 1
                if src not in source_weights:
                    source_weights[src] = []
                source_weights[src].append(a.weight)
                source_pnl[src] = source_pnl.get(src, 0.0) + pnl * a.weight

        summary = AttributionSummary(
            total_decisions=len(self._history),
            source_win_rates={
                src: source_wins.get(src, 0) / cnt
                for src, cnt in source_total.items()
                if cnt > 0
            },
            source_avg_weights={
                src: sum(ws) / len(ws)
                for src, ws in source_weights.items()
            },
            source_pnl_contribution=source_pnl,
        )

        log.info(
            "attribution_summary",
            total_decisions=summary.total_decisions,
            sources_tracked=len(summary.source_avg_weights),
        )

        return summary

    @staticmethod
    def _infer_direction(
        text: str, keywords: list[str], action: Action
    ) -> float:
        """Infer if a data source pushed bullish or bearish."""
        # Simple heuristic: if the signal is BUY, sources are bullish; SELL = bearish
        if action == Action.BUY:
            return 1.0
        if action == Action.SELL:
            return -1.0
        return 0.0
