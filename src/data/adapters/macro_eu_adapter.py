"""Europe macroeconomic data adapter using yfinance and FRED."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from config.settings import Settings, get_settings
from src.core.constants import LLM_MAX_RETRIES
from src.core.exceptions import DataFetchError, RateLimitError
from src.core.logging import get_logger
from src.core.types import MacroData

log = get_logger(__name__)

# FRED series IDs for Europe
_SERIES_ECB_RATE = "ECBMRRFR"        # ECB Main Refinancing Rate
_SERIES_BOE_RATE = "BOERUKM"         # Bank of England Official Bank Rate
_SERIES_DE_10Y = "IRLTLT01DEM156N"   # Germany 10-year government bond yield

# yfinance tickers for FX rates
_EURUSD_TICKER = "EURUSD=X"
_GBPUSD_TICKER = "GBPUSD=X"

_ALL_SERIES: list[str] = [_SERIES_ECB_RATE, _SERIES_BOE_RATE, _SERIES_DE_10Y]

_LOOKBACK_DAYS = 400


class MacroEUAdapter:
    """Fetches European macroeconomic indicators from FRED and yfinance.

    Retrieves:
    * ECB main refinancing rate (from FRED)
    * Bank of England official bank rate (from FRED)
    * Germany 10-year government bond yield (from FRED)
    * EUR/USD exchange rate (from yfinance)
    * GBP/USD exchange rate (from yfinance)

    ``fredapi.Fred`` and ``yfinance`` are synchronous, so all API calls
    are wrapped with ``asyncio.to_thread`` to avoid blocking the event loop.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        api_key: str = self._settings.fred_api_key.get_secret_value()
        if not api_key:
            msg = "fred_api_key is not configured in settings / .env"
            raise DataFetchError(msg)
        from fredapi import Fred
        self._fred: Fred = Fred(api_key=api_key)
        self._max_retries: int = LLM_MAX_RETRIES
        log.info("macro_eu_adapter_initialized")

    # -- Internal helpers --------------------------------------------------

    def _fetch_series_sync(self, series_id: str) -> float | None:
        """Fetch the latest non-NaN value for a single FRED series.

        Returns ``None`` when data is unavailable rather than raising.
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=_LOOKBACK_DAYS)
        try:
            series = self._fred.get_series(
                series_id,
                observation_start=start.strftime("%Y-%m-%d"),
                observation_end=end.strftime("%Y-%m-%d"),
            )
            if series is None or series.empty:
                log.warning("fred_series_empty", series_id=series_id)
                return None
            clean = series.dropna()
            if clean.empty:
                log.warning("fred_series_all_nan", series_id=series_id)
                return None
            return float(clean.iloc[-1])
        except Exception as exc:
            log.warning(
                "fred_series_fetch_failed",
                series_id=series_id,
                error=str(exc),
            )
            return None

    def _fetch_fx_rate_sync(self, ticker_symbol: str) -> float | None:
        """Fetch the latest FX rate via yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                log.warning("yfinance_fx_empty", ticker=ticker_symbol)
                return None
            return float(hist.iloc[-1]["Close"])
        except Exception as exc:
            log.warning(
                "yfinance_fx_fetch_failed",
                ticker=ticker_symbol,
                error=str(exc),
            )
            return None

    def _fetch_all_sync(self) -> MacroData:
        """Synchronous aggregate fetch -- called via asyncio.to_thread."""
        ecb_rate = self._fetch_series_sync(_SERIES_ECB_RATE)
        boe_rate = self._fetch_series_sync(_SERIES_BOE_RATE)
        de_10y_yield = self._fetch_series_sync(_SERIES_DE_10Y)
        eurusd = self._fetch_fx_rate_sync(_EURUSD_TICKER)
        gbpusd = self._fetch_fx_rate_sync(_GBPUSD_TICKER)

        macro = MacroData(
            ecb_rate=ecb_rate,
            boe_rate=boe_rate,
            de_10y_yield=de_10y_yield,
            eurusd=eurusd,
            gbpusd=gbpusd,
        )

        log.info(
            "macro_eu_fetch_complete",
            ecb_rate=ecb_rate,
            boe_rate=boe_rate,
            de_10y_yield=de_10y_yield,
            eurusd=eurusd,
            gbpusd=gbpusd,
        )
        return macro

    # -- Public API --------------------------------------------------------

    async def fetch_macro(self) -> MacroData:
        """Fetch European macroeconomic data with retry logic.

        Returns a :class:`MacroData` populated with ``ecb_rate``,
        ``boe_rate``, ``de_10y_yield``, ``eurusd``, and ``gbpusd``.
        Fields that cannot be retrieved are left as ``None``.

        Raises:
            DataFetchError: If all retry attempts are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug("macro_eu_fetch_attempt", attempt=attempt)
                macro = await asyncio.to_thread(self._fetch_all_sync)
                log.info(
                    "macro_eu_fetch_success",
                    attempt=attempt,
                    has_ecb_rate=macro.ecb_rate is not None,
                    has_boe_rate=macro.boe_rate is not None,
                    has_de_10y=macro.de_10y_yield is not None,
                    has_eurusd=macro.eurusd is not None,
                    has_gbpusd=macro.gbpusd is not None,
                )
                return macro

            except RateLimitError:
                raise

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "macro_eu_fetch_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0 * attempt)

        msg = f"Europe macro data fetch failed after {self._max_retries} attempts"
        raise DataFetchError(
            msg,
            context={
                "series": _ALL_SERIES,
                "last_error": str(last_exc),
            },
        )
