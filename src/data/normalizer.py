"""Data normalizer — converts raw adapter data into standardized MarketSnapshots."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from uuid_extensions import uuid7

from src.core.logging import get_logger
from src.core.types import (
    MacroData,
    Market,
    MarketSnapshot,
    NewsItem,
    SymbolData,
)

log = get_logger(__name__)

FORWARD_FILL_MAX_HOURS = 3


class DataNormalizer:
    """Convert market-specific raw data into standardized MarketSnapshot objects.

    Responsibilities:
    1. Timestamp -> UTC conversion
    2. Market-specific raw schema -> SymbolData unification
    3. News/sentiment merge
    4. MarketSnapshot creation & snapshot_id issuance (uuid7)

    Rules:
    - Missing data: forward-fill up to 3 hours
    - Beyond 3 hours: exclude symbol from snapshot + warning log
    """

    def __init__(self) -> None:
        self._last_snapshots: dict[str, MarketSnapshot] = {}
        self._last_symbol_data: dict[str, dict[str, SymbolData]] = {}
        self._last_symbol_times: dict[str, dict[str, datetime]] = {}

    async def create_snapshot(
        self,
        market: Market,
        symbols: dict[str, SymbolData],
        macro: MacroData,
        news: list[NewsItem] | None = None,
    ) -> MarketSnapshot:
        """Create a normalized MarketSnapshot from raw adapter data.

        Args:
            market: Target market
            symbols: Raw symbol data from adapter
            macro: Macroeconomic data
            news: News items (empty list if unavailable)

        Returns:
            Normalized and validated MarketSnapshot
        """
        now = datetime.now(timezone.utc)
        market_key = market.value

        # Forward-fill missing symbols from last known data
        filled_symbols = self._forward_fill(market_key, symbols, now)

        # Update cache
        self._last_symbol_data[market_key] = filled_symbols
        for sym in symbols:
            if market_key not in self._last_symbol_times:
                self._last_symbol_times[market_key] = {}
            self._last_symbol_times[market_key][sym] = now

        snapshot = MarketSnapshot(
            snapshot_id=str(uuid7()),
            timestamp=now,
            market=market,
            symbols=filled_symbols,
            macro=macro,
            news=news or [],
            metadata={
                "fresh_symbols": len(symbols),
                "filled_symbols": len(filled_symbols) - len(symbols),
                "excluded_symbols": 0,
            },
        )

        self._last_snapshots[market_key] = snapshot
        log.info(
            "snapshot_created",
            market=market_key,
            snapshot_id=snapshot.snapshot_id,
            symbol_count=len(filled_symbols),
            fresh=len(symbols),
            news_count=len(snapshot.news),
        )

        return snapshot

    def _forward_fill(
        self,
        market_key: str,
        fresh_symbols: dict[str, SymbolData],
        now: datetime,
    ) -> dict[str, SymbolData]:
        """Forward-fill missing symbols from last known data.

        Symbols older than FORWARD_FILL_MAX_HOURS are excluded.
        """
        result = dict(fresh_symbols)
        cutoff = now - timedelta(hours=FORWARD_FILL_MAX_HOURS)
        excluded_count = 0

        last_data = self._last_symbol_data.get(market_key, {})
        last_times = self._last_symbol_times.get(market_key, {})

        for sym, data in last_data.items():
            if sym not in result:
                last_time = last_times.get(sym)
                if last_time and last_time >= cutoff:
                    result[sym] = data
                    log.debug(
                        "symbol_forward_filled",
                        symbol=sym,
                        age_minutes=(now - last_time).total_seconds() / 60,
                    )
                else:
                    excluded_count += 1
                    log.warning(
                        "symbol_excluded_stale",
                        symbol=sym,
                        last_seen=str(last_time) if last_time else "never",
                    )

        if excluded_count > 0:
            log.warning("symbols_excluded_from_snapshot", count=excluded_count)

        return result
