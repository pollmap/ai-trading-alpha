"""Europe (EURONEXT) market data adapter using yfinance."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml
import yfinance as yf

from config.settings import Settings, get_settings
from src.core.constants import LLM_MAX_RETRIES
from src.core.exceptions import DataFetchError
from src.core.interfaces import BaseMarketDataAdapter
from src.core.logging import get_logger
from src.core.types import Market, SymbolData

log = get_logger(__name__)

_MARKETS_YAML = Path(__file__).resolve().parents[3] / "config" / "markets.yaml"

_DEFAULT_SYMBOLS: list[str] = [
    "MC.PA",    # LVMH (Paris)
    "ASML.AS",  # ASML (Amsterdam)
    "OR.PA",    # L'Oreal (Paris)
    "SAN.PA",   # Sanofi (Paris)
    "AIR.PA",   # Airbus (Paris)
]


def _load_euronext_symbols() -> list[str]:
    """Load EURONEXT symbol list from config/markets.yaml with fallback to defaults."""
    try:
        with _MARKETS_YAML.open() as fh:
            config: dict[str, object] = yaml.safe_load(fh)
        euronext_section = config.get("EURONEXT", {})
        symbols: list[str] = euronext_section.get("symbols", [])  # type: ignore[union-attr]
        if symbols:
            return symbols
    except Exception as exc:
        log.warning("euronext_config_load_fallback", error=str(exc))
    return list(_DEFAULT_SYMBOLS)


def _fetch_yfinance_batch(symbols: list[str]) -> dict[str, SymbolData]:
    """Synchronous yfinance fetch -- will be wrapped with asyncio.to_thread.

    Downloads the latest 1-day OHLCV bar for every symbol in a single batch
    request. Symbols that fail individually are logged and skipped.
    """
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
            info = ticker.info or {}

            result[symbol] = SymbolData(
                symbol=symbol,
                market=Market.EURONEXT,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                currency="EUR",
                per=info.get("trailingPE"),
                pbr=info.get("priceToBook"),
                market_cap=info.get("marketCap"),
                extra={
                    "dividend_yield": info.get("dividendYield"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                },
            )
        except Exception as exc:
            log.warning(
                "yfinance_symbol_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )
            continue

    return result


class EuronextAdapter(BaseMarketDataAdapter):
    """Europe (EURONEXT) stock market data adapter using yfinance for OHLCV data.

    yfinance is a synchronous library, so all calls are wrapped with
    ``asyncio.to_thread`` to keep the event loop non-blocking.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._symbols: list[str] = _load_euronext_symbols()
        self._max_retries: int = LLM_MAX_RETRIES
        log.info(
            "euronext_adapter_initialized",
            symbol_count=len(self._symbols),
            symbols=self._symbols,
        )

    @property
    def symbols(self) -> list[str]:
        """Return the configured EURONEXT symbol list."""
        return list(self._symbols)

    async def fetch_latest(self) -> dict[str, SymbolData]:
        """Fetch the latest OHLCV data for all configured EURONEXT symbols.

        Retries up to 3 times on failure, then raises DataFetchError.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug(
                    "euronext_fetch_attempt",
                    attempt=attempt,
                    symbol_count=len(self._symbols),
                )
                result = await asyncio.to_thread(
                    _fetch_yfinance_batch, self._symbols,
                )
                if not result:
                    msg = "yfinance returned data for zero EURONEXT symbols"
                    raise DataFetchError(msg)

                log.info(
                    "euronext_fetch_success",
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
                    "euronext_fetch_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0 * attempt)

        msg = f"EURONEXT market data fetch failed after {self._max_retries} attempts"
        raise DataFetchError(
            msg,
            context={
                "symbols": self._symbols,
                "last_error": str(last_exc),
            },
        )
