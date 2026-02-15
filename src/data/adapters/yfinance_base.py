"""Shared base class for all yfinance-backed market data adapters.

Eliminates duplication across JPX, SSE, HKEX, EURONEXT, LSE, BOND, and
COMMODITIES adapters by extracting the common yfinance fetch + retry logic
into a configurable base.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from config.settings import Settings, get_settings
from src.core.constants import LLM_MAX_RETRIES
from src.core.exceptions import DataFetchError
from src.core.interfaces import BaseMarketDataAdapter
from src.core.logging import get_logger
from src.core.types import Market, SymbolData

log = get_logger(__name__)

_MARKETS_YAML = Path(__file__).resolve().parents[3] / "config" / "markets.yaml"


@dataclass(frozen=True)
class YFinanceAdapterConfig:
    """Configuration for a yfinance-based market adapter."""

    market: Market
    currency: str
    yaml_section: str
    default_symbols: list[str]
    # Whether to pull ticker.info for fundamentals
    use_info: bool = True
    # Info field names (None = skip)
    per_field: str | None = "trailingPE"
    pbr_field: str | None = "priceToBook"
    market_cap_field: str | None = "marketCap"
    # Extra fields extracted from ticker.info: {output_key: info_key}
    info_extra_fields: dict[str, str] = field(default_factory=dict)
    # Static extra fields (not from info): {key: value}
    static_extra: dict[str, Any] = field(default_factory=dict)


def _load_symbols_from_yaml(yaml_section: str, defaults: list[str]) -> list[str]:
    """Load symbol list from config/markets.yaml with fallback to defaults."""
    try:
        with _MARKETS_YAML.open() as fh:
            config: dict[str, object] = yaml.safe_load(fh)
        section = config.get(yaml_section, {})
        symbols: list[str] = section.get("symbols", [])  # type: ignore[union-attr]
        if symbols:
            return symbols
    except Exception as exc:
        log.warning("config_load_fallback", section=yaml_section, error=str(exc))
    return list(defaults)


def _yfinance_fetch_batch(
    symbols: list[str],
    config: YFinanceAdapterConfig,
) -> dict[str, SymbolData]:
    """Synchronous yfinance fetch — wrapped with asyncio.to_thread by callers.

    Downloads the latest 1-day OHLCV bar for every symbol in a single batch
    request.  Symbols that fail individually are logged and skipped.
    """
    import yfinance as yf

    joined = " ".join(symbols)
    tickers = yf.Tickers(joined)
    result: dict[str, SymbolData] = {}

    for symbol in symbols:
        try:
            ticker = tickers.tickers.get(symbol)
            if ticker is None:
                log.warning("yfinance_ticker_not_found", symbol=symbol)
                continue

            hist = ticker.history(period="1d")
            if hist.empty:
                log.warning("yfinance_empty_history", symbol=symbol)
                continue

            row = hist.iloc[-1]

            # Extract fundamentals from ticker.info when configured
            info: dict[str, Any] = {}
            if config.use_info:
                info = ticker.info or {}

            per = info.get(config.per_field) if config.per_field else None
            pbr = info.get(config.pbr_field) if config.pbr_field else None
            market_cap = info.get(config.market_cap_field) if config.market_cap_field else None

            # Build extra dict from info fields + static fields
            extra: dict[str, Any] = {}
            for out_key, info_key in config.info_extra_fields.items():
                extra[out_key] = info.get(info_key)
            extra.update(config.static_extra)

            result[symbol] = SymbolData(
                symbol=symbol,
                market=config.market,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                currency=config.currency,
                per=per,
                pbr=pbr,
                market_cap=market_cap,
                extra=extra,
            )
        except Exception as exc:
            log.warning(
                "yfinance_symbol_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )
            continue

    return result


class YFinanceBaseAdapter(BaseMarketDataAdapter):
    """Base adapter for markets that use yfinance for OHLCV data.

    yfinance is a synchronous library, so all calls are wrapped with
    ``asyncio.to_thread`` to keep the event loop non-blocking.

    Subclasses only need to supply a ``YFinanceAdapterConfig`` via
    ``super().__init__(config=...)``.
    """

    def __init__(
        self,
        config: YFinanceAdapterConfig,
        settings: Settings | None = None,
    ) -> None:
        self._config = config
        self._settings: Settings = settings or get_settings()
        self._symbols: list[str] = _load_symbols_from_yaml(
            config.yaml_section, config.default_symbols,
        )
        self._max_retries: int = LLM_MAX_RETRIES
        log.info(
            "adapter_initialized",
            market=config.market.value,
            symbol_count=len(self._symbols),
            symbols=self._symbols,
        )

    @property
    def symbols(self) -> list[str]:
        """Return the configured symbol list."""
        return list(self._symbols)

    async def fetch_latest(self) -> dict[str, SymbolData]:
        """Fetch latest OHLCV data for all configured symbols.

        Retries up to ``LLM_MAX_RETRIES`` times, then raises DataFetchError.
        """
        market_name = self._config.market.value
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug(
                    "fetch_attempt",
                    market=market_name,
                    attempt=attempt,
                    symbol_count=len(self._symbols),
                )
                result = await asyncio.to_thread(
                    _yfinance_fetch_batch, self._symbols, self._config,
                )
                if not result:
                    msg = f"yfinance returned data for zero {market_name} symbols"
                    raise DataFetchError(msg)

                log.info(
                    "fetch_success",
                    market=market_name,
                    fetched=len(result),
                    total=len(self._symbols),
                    attempt=attempt,
                )
                return result

            except DataFetchError:
                raise

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "fetch_retry",
                    market=market_name,
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0 * attempt)

        msg = f"{market_name} market data fetch failed after {self._max_retries} attempts"
        raise DataFetchError(
            msg,
            context={
                "symbols": self._symbols,
                "last_error": str(last_exc),
            },
        )
