"""Cryptocurrency data adapter using Binance API and CCXT fallback."""

from __future__ import annotations

import asyncio
from pathlib import Path

import ccxt.async_support as ccxt_async
import httpx
import yaml

from config.settings import Settings, get_settings
from src.core.constants import LLM_MAX_RETRIES
from src.core.exceptions import DataFetchError
from src.core.interfaces import BaseMarketDataAdapter
from src.core.logging import get_logger
from src.core.types import Market, SymbolData

log = get_logger(__name__)

_MARKETS_YAML = Path(__file__).resolve().parents[3] / "config" / "markets.yaml"

# Alternative.me Fear & Greed Index endpoint.
_FEAR_GREED_URL = "https://api.alternative.me/fng/"

# Exponential backoff parameters.
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MAX_SECONDS = 30.0


def _load_crypto_symbols() -> list[str]:
    """Load crypto symbol list from config/markets.yaml."""
    with _MARKETS_YAML.open() as fh:
        config: dict[str, object] = yaml.safe_load(fh)
    crypto_section = config.get("CRYPTO", {})
    symbols: list[str] = crypto_section.get("symbols", [])  # type: ignore[union-attr]
    if not symbols:
        msg = "No CRYPTO symbols found in markets.yaml"
        raise DataFetchError(msg, context={"path": str(_MARKETS_YAML)})
    return symbols


def _backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay capped at ``_BACKOFF_MAX_SECONDS``."""
    delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    return min(delay, _BACKOFF_MAX_SECONDS)


class CryptoAdapter(BaseMarketDataAdapter):
    """Cryptocurrency data adapter.

    Uses ``ccxt.binance`` (async) as the primary source for OHLCV candles.
    An optional ``fetch_fear_greed`` method retrieves the Crypto Fear & Greed
    Index from alternative.me via ``httpx``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._symbols: list[str] = _load_crypto_symbols()
        self._max_retries: int = LLM_MAX_RETRIES

        api_key: str = self._settings.binance_api_key.get_secret_value()
        api_secret: str = self._settings.binance_api_secret.get_secret_value()

        exchange_config: dict[str, object] = {
            "enableRateLimit": True,
        }
        if api_key and api_secret:
            exchange_config["apiKey"] = api_key
            exchange_config["secret"] = api_secret

        self._exchange: ccxt_async.binance = ccxt_async.binance(exchange_config)
        log.info(
            "crypto_adapter_initialized",
            symbol_count=len(self._symbols),
            symbols=self._symbols,
            authenticated=bool(api_key),
        )

    @property
    def symbols(self) -> list[str]:
        """Return the configured crypto symbol list."""
        return list(self._symbols)

    # ── Internal helpers ─────────────────────────────────────────

    async def _fetch_symbol_ohlcv(self, symbol: str) -> SymbolData | None:
        """Fetch the latest 1-day OHLCV candle for a single symbol.

        CCXT expects symbols in ``BASE/QUOTE`` format (e.g. ``BTC/USDT``),
        while the config stores them as ``BTCUSDT``.  This method handles the
        conversion internally.
        """
        # Convert "BTCUSDT" -> "BTC/USDT".
        ccxt_symbol = symbol.replace("USDT", "/USDT")

        try:
            candles: list[list[float]] = await self._exchange.fetch_ohlcv(
                ccxt_symbol,
                timeframe="1d",
                limit=1,
            )
            if not candles:
                log.warning("crypto_empty_candles", symbol=symbol)
                return None

            # CCXT candle format: [timestamp, open, high, low, close, volume]
            candle = candles[-1]
            return SymbolData(
                symbol=symbol,
                market=Market.CRYPTO,
                open=float(candle[1]),
                high=float(candle[2]),
                low=float(candle[3]),
                close=float(candle[4]),
                volume=float(candle[5]),
                currency="USDT",
            )
        except Exception as exc:
            log.warning(
                "crypto_symbol_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )
            return None

    # ── Public API ───────────────────────────────────────────────

    async def fetch_latest(self) -> dict[str, SymbolData]:
        """Fetch the latest OHLCV data for all configured crypto symbols.

        Uses exponential backoff on retries.  Individual symbol failures are
        logged and skipped; a ``DataFetchError`` is raised only when no
        symbols could be fetched after exhausting retries.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug(
                    "crypto_fetch_attempt",
                    attempt=attempt,
                    symbol_count=len(self._symbols),
                )

                tasks = [
                    self._fetch_symbol_ohlcv(symbol)
                    for symbol in self._symbols
                ]
                results: list[SymbolData | None] = await asyncio.gather(
                    *tasks, return_exceptions=False,
                )

                data: dict[str, SymbolData] = {
                    sd.symbol: sd for sd in results if sd is not None
                }

                if not data:
                    msg = "ccxt returned data for zero symbols"
                    raise DataFetchError(msg)

                log.info(
                    "crypto_fetch_success",
                    fetched=len(data),
                    total=len(self._symbols),
                    attempt=attempt,
                )
                return data

            except DataFetchError:
                raise

            except Exception as exc:
                last_exc = exc
                delay = _backoff_delay(attempt)
                log.warning(
                    "crypto_fetch_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    backoff_seconds=delay,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)

        msg = f"Crypto market data fetch failed after {self._max_retries} attempts"
        raise DataFetchError(
            msg,
            context={
                "symbols": self._symbols,
                "last_error": str(last_exc),
            },
        )

    async def fetch_fear_greed(self) -> float:
        """Fetch the current Crypto Fear & Greed Index (0-100).

        Source: https://alternative.me/crypto/fear-and-greed-index/

        Returns:
            The index value as a float (0 = extreme fear, 100 = extreme greed).

        Raises:
            DataFetchError: If the API call fails after retries.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug("fear_greed_fetch_attempt", attempt=attempt)
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        _FEAR_GREED_URL,
                        params={"limit": 1, "format": "json"},
                    )
                    response.raise_for_status()

                payload: dict[str, object] = response.json()
                data_list: list[dict[str, str]] = payload.get("data", [])  # type: ignore[assignment]
                if not data_list:
                    msg = "Fear & Greed API returned empty data"
                    raise DataFetchError(msg)

                value = float(data_list[0]["value"])
                log.info("fear_greed_fetch_success", value=value, attempt=attempt)
                return value

            except DataFetchError:
                raise

            except Exception as exc:
                last_exc = exc
                delay = _backoff_delay(attempt)
                log.warning(
                    "fear_greed_fetch_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    backoff_seconds=delay,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)

        msg = f"Fear & Greed index fetch failed after {self._max_retries} attempts"
        raise DataFetchError(
            msg,
            context={"url": _FEAR_GREED_URL, "last_error": str(last_exc)},
        )

    async def close(self) -> None:
        """Close the underlying CCXT exchange connection."""
        try:
            await self._exchange.close()
            log.info("crypto_adapter_closed")
        except Exception as exc:
            log.warning("crypto_adapter_close_error", error=str(exc))
