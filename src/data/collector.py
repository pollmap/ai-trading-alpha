"""Unified data collection layer — dispatches to market-specific adapters.

Creates MarketSnapshot objects by fetching symbol data from the appropriate
adapter for each market and running it through the DataNormalizer.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Any

import yaml

from src.core.interfaces import BaseMarketDataAdapter
from src.core.logging import get_logger
from src.core.types import MacroData, Market, MarketSnapshot
from src.data.normalizer import DataNormalizer

log = get_logger(__name__)

_MARKETS_YAML = Path(__file__).resolve().parents[2] / "config" / "markets.yaml"

# Market -> (module_path, class_name)
_ADAPTER_REGISTRY: dict[str, tuple[str, str]] = {
    "KRX": ("src.data.adapters.krx_adapter", "KRXAdapter"),
    "US": ("src.data.adapters.us_adapter", "USAdapter"),
    "CRYPTO": ("src.data.adapters.crypto_adapter", "CryptoAdapter"),
    "JPX": ("src.data.adapters.jpx_adapter", "JPXAdapter"),
    "SSE": ("src.data.adapters.sse_adapter", "SSEAdapter"),
    "HKEX": ("src.data.adapters.hkex_adapter", "HKEXAdapter"),
    "EURONEXT": ("src.data.adapters.euronext_adapter", "EuronextAdapter"),
    "LSE": ("src.data.adapters.lse_adapter", "LSEAdapter"),
    "BOND": ("src.data.adapters.bond_adapter", "BondAdapter"),
    "COMMODITIES": ("src.data.adapters.commodities_adapter", "CommoditiesAdapter"),
}

# Macro adapter registry
_MACRO_REGISTRY: dict[str, tuple[str, str]] = {
    "KRX": ("src.data.adapters.macro_kr_adapter", "MacroKRAdapter"),
    "US": ("src.data.adapters.macro_us_adapter", "MacroUSAdapter"),
    "JPX": ("src.data.adapters.macro_jp_adapter", "MacroJPAdapter"),
    "SSE": ("src.data.adapters.macro_cn_adapter", "MacroCNAdapter"),
    "EURONEXT": ("src.data.adapters.macro_eu_adapter", "MacroEUAdapter"),
}


class DataCollector:
    """Unified data collection — fetches market data and creates snapshots.

    Lazily initializes adapters for each market. Adapters that fail to import
    (due to missing dependencies) are silently skipped with a warning.

    Usage::

        collector = DataCollector()
        snapshots = await collector.collect_all([Market.US, Market.CRYPTO])
        # snapshots: dict[Market, MarketSnapshot]
    """

    def __init__(self) -> None:
        self._adapters: dict[Market, BaseMarketDataAdapter] = {}
        self._macro_adapters: dict[Market, Any] = {}
        self._normalizer = DataNormalizer()
        self._initialized_markets: set[str] = set()

    def _ensure_adapter(self, market: Market) -> BaseMarketDataAdapter | None:
        """Lazily initialize and return adapter for a market."""
        if market in self._adapters:
            return self._adapters[market]

        if market.value in self._initialized_markets:
            return None  # Already tried and failed

        self._initialized_markets.add(market.value)

        entry = _ADAPTER_REGISTRY.get(market.value)
        if entry is None:
            log.warning("no_adapter_registered", market=market.value)
            return None

        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            adapter_cls = getattr(module, class_name)
            adapter = adapter_cls()
            self._adapters[market] = adapter
            log.info("adapter_initialized", market=market.value, adapter=class_name)
            return adapter
        except Exception as exc:
            log.warning(
                "adapter_init_failed",
                market=market.value,
                module=module_path,
                error=str(exc),
            )
            return None

    def _ensure_macro_adapter(self, market: Market) -> Any | None:
        """Lazily initialize macro adapter for a market."""
        if market in self._macro_adapters:
            return self._macro_adapters[market]

        entry = _MACRO_REGISTRY.get(market.value)
        if entry is None:
            return None

        module_path, class_name = entry
        try:
            module = importlib.import_module(module_path)
            adapter_cls = getattr(module, class_name)
            adapter = adapter_cls()
            self._macro_adapters[market] = adapter
            return adapter
        except Exception:
            return None

    async def collect_snapshot(self, market: Market) -> MarketSnapshot | None:
        """Collect data for a single market and create a MarketSnapshot.

        Returns None if the adapter is unavailable or the fetch fails.
        """
        adapter = self._ensure_adapter(market)
        if adapter is None:
            log.warning("snapshot_skipped_no_adapter", market=market.value)
            return None

        try:
            symbols = await adapter.fetch_latest()
            if not symbols:
                log.warning("snapshot_skipped_empty_data", market=market.value)
                return None

            # Try macro data (optional, don't fail if unavailable)
            macro = await self._fetch_macro(market)

            snapshot = await self._normalizer.create_snapshot(
                market=market,
                symbols=symbols,
                macro=macro,
            )

            log.info(
                "snapshot_collected",
                market=market.value,
                symbols=len(symbols),
                snapshot_id=snapshot.snapshot_id,
            )
            return snapshot

        except Exception as exc:
            log.error(
                "snapshot_collection_failed",
                market=market.value,
                error=str(exc),
            )
            return None

    async def _fetch_macro(self, market: Market) -> MacroData:
        """Fetch macro data for a market, returning empty MacroData on failure."""
        macro_adapter = self._ensure_macro_adapter(market)
        if macro_adapter is None:
            return MacroData()

        try:
            if hasattr(macro_adapter, "fetch_latest"):
                return await macro_adapter.fetch_latest()
        except Exception as exc:
            log.debug("macro_fetch_failed", market=market.value, error=str(exc))

        return MacroData()

    async def collect_all(
        self, markets: list[Market],
    ) -> dict[Market, MarketSnapshot]:
        """Collect snapshots for all markets in parallel.

        Markets whose adapters are unavailable or whose fetch fails
        are silently skipped.
        """
        tasks: dict[Market, asyncio.Task[MarketSnapshot | None]] = {}
        for market in markets:
            tasks[market] = asyncio.create_task(
                self.collect_snapshot(market),
                name=f"collect_{market.value}",
            )

        results: dict[Market, MarketSnapshot] = {}
        for market, task in tasks.items():
            try:
                snapshot = await task
                if snapshot is not None:
                    results[market] = snapshot
            except Exception as exc:
                log.error(
                    "collect_all_market_failed",
                    market=market.value,
                    error=str(exc),
                )

        log.info(
            "collect_all_complete",
            requested=len(markets),
            collected=len(results),
            markets=[m.value for m in results],
        )
        return results

    @property
    def available_markets(self) -> list[str]:
        """Return list of markets with successfully initialized adapters."""
        return [m.value for m in self._adapters]
