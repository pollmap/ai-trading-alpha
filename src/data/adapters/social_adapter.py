"""Social sentiment adapter — Reddit, Twitter/X, StockTwits aggregation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import aiohttp

from src.core.logging import get_logger

log = get_logger(__name__)


class SentimentSource(str, Enum):
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    TWITTER = "twitter"
    LUNARCRUSH = "lunarcrush"
    AGGREGATED = "aggregated"


@dataclass
class SocialMention:
    """Single social media mention with sentiment."""
    source: SentimentSource
    timestamp: datetime
    text: str
    sentiment_score: float  # -1.0 to +1.0
    engagement: int = 0  # likes + comments + retweets
    author_credibility: float = 0.5  # 0-1


@dataclass
class SocialSentimentSnapshot:
    """Aggregated social sentiment for a symbol or market."""
    symbol: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mentions: list[SocialMention] = field(default_factory=list)
    mention_count_1h: int = 0
    mention_count_24h: int = 0
    avg_sentiment: float = 0.0  # -1 to +1
    sentiment_momentum: float = 0.0  # change in sentiment over time
    bullish_ratio: float = 0.5  # ratio of bullish mentions
    social_volume_change_pct: float = 0.0  # % change in social volume vs baseline

    def to_prompt_line(self) -> str:
        """Compact summary for LLM prompt injection."""
        sentiment_label = "Bullish" if self.avg_sentiment > 0.1 else "Bearish" if self.avg_sentiment < -0.1 else "Neutral"
        parts: list[str] = [f"{self.symbol}:"]
        parts.append(f"Social={sentiment_label}({self.avg_sentiment:+.2f})")
        parts.append(f"Mentions24h={self.mention_count_24h}")
        if self.social_volume_change_pct != 0:
            parts.append(f"VolChg={self.social_volume_change_pct:+.0f}%")
        parts.append(f"Bull%={self.bullish_ratio:.0%}")
        return " ".join(parts)


@dataclass
class MarketSocialSnapshot:
    """Aggregated social data across all symbols for a market."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbols: dict[str, SocialSentimentSnapshot] = field(default_factory=dict)
    overall_sentiment: float = 0.0
    trending_symbols: list[str] = field(default_factory=list)

    def to_prompt_summary(self) -> str:
        lines: list[str] = ["=== Social Sentiment ==="]
        lines.append(f"Overall Market Sentiment: {self.overall_sentiment:+.2f}")
        if self.trending_symbols:
            lines.append(f"Trending: {', '.join(self.trending_symbols[:5])}")
        for sym, snap in self.symbols.items():
            lines.append(snap.to_prompt_line())
        return "\n".join(lines)


class SocialSentimentAdapter:
    """Fetch social sentiment data from various APIs.

    In production, connects to:
    - Reddit API (r/wallstreetbets, r/stocks, r/cryptocurrency)
    - LunarCrush API (crypto social metrics)
    - StockTwits API

    Provides structured interface with graceful degradation when APIs unavailable.
    """

    def __init__(
        self,
        reddit_client_id: str = "",
        reddit_client_secret: str = "",
        lunarcrush_api_key: str = "",
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._reddit_id: str = reddit_client_id
        self._reddit_secret: str = reddit_client_secret
        self._lunar_key: str = lunarcrush_api_key
        self._session: aiohttp.ClientSession | None = session
        self._owns_session: bool = False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def fetch_symbol_sentiment(self, symbol: str) -> SocialSentimentSnapshot:
        """Fetch social sentiment for a single symbol.

        Currently returns a default snapshot. In production, aggregates from
        multiple social data sources.
        """
        snapshot = SocialSentimentSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
        )
        log.debug("social_sentiment_fetched", symbol=symbol)
        return snapshot

    async def fetch_market_sentiment(
        self, symbols: list[str]
    ) -> MarketSocialSnapshot:
        """Fetch social sentiment for all symbols in a market."""
        tasks = [self.fetch_symbol_sentiment(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        symbol_snapshots: dict[str, SocialSentimentSnapshot] = {}
        for result in results:
            if isinstance(result, SocialSentimentSnapshot):
                symbol_snapshots[result.symbol] = result

        # Calculate overall sentiment
        sentiments = [s.avg_sentiment for s in symbol_snapshots.values()]
        overall = sum(sentiments) / len(sentiments) if sentiments else 0.0

        # Find trending (highest mention count change)
        trending = sorted(
            symbol_snapshots.keys(),
            key=lambda s: symbol_snapshots[s].social_volume_change_pct,
            reverse=True,
        )[:5]

        market_snap = MarketSocialSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbols=symbol_snapshots,
            overall_sentiment=overall,
            trending_symbols=trending,
        )
        log.info(
            "market_social_sentiment_fetched",
            symbol_count=len(symbol_snapshots),
            overall_sentiment=overall,
        )
        return market_snap
