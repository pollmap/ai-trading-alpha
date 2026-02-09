"""Japan macroeconomic data adapter using yfinance and FRED."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import yfinance as yf
from fredapi import Fred

from config.settings import Settings, get_settings
from src.core.constants import LLM_MAX_RETRIES
from src.core.exceptions import DataFetchError, RateLimitError
from src.core.logging import get_logger
from src.core.types import MacroData

log = get_logger(__name__)

# FRED series IDs for Japan
_SERIES_JP_IR = "IRSTCI01JPM156N"  # Japan short-term interest rate (proxy for base rate)
_SERIES_JP_10Y = "IRLTLT01JPM156N"  # Japan long-term government bond yield (10Y)

# yfinance ticker for USD/JPY exchange rate
_USDJPY_TICKER = "JPY=X"

_ALL_SERIES: list[str] = [_SERIES_JP_IR, _SERIES_JP_10Y]

_LOOKBACK_DAYS = 400


class MacroJPAdapter:
    """Fetches Japan macroeconomic indicators from FRED and yfinance.

    Retrieves:
    * BOJ base rate (proxy via short-term interest rate from FRED)
    * Japan 10-year government bond yield (from FRED)
    * USD/JPY exchange rate (from yfinance)

    ``fredapi.Fred`` and ``yfinance`` are synchronous, so all API calls
    are wrapped with ``asyncio.to_thread`` to avoid blocking the event loop.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        api_key: str = self._settings.fred_api_key.get_secret_value()
        if not api_key:
            msg = "fred_api_key is not configured in settings / .env"
            raise DataFetchError(msg)
        self._fred: Fred = Fred(api_key=api_key)
        self._max_retries: int = LLM_MAX_RETRIES
        log.info("macro_jp_adapter_initialized")

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

    def _fetch_usdjpy_sync(self) -> float | None:
        """Fetch the latest USD/JPY exchange rate via yfinance."""
        try:
            ticker = yf.Ticker(_USDJPY_TICKER)
            hist = ticker.history(period="1d")
            if hist.empty:
                log.warning("yfinance_usdjpy_empty")
                return None
            return float(hist.iloc[-1]["Close"])
        except Exception as exc:
            log.warning("yfinance_usdjpy_fetch_failed", error=str(exc))
            return None

    def _fetch_all_sync(self) -> MacroData:
        """Synchronous aggregate fetch -- called via asyncio.to_thread."""
        jp_base_rate = self._fetch_series_sync(_SERIES_JP_IR)
        jp_10y_yield = self._fetch_series_sync(_SERIES_JP_10Y)
        usdjpy = self._fetch_usdjpy_sync()

        macro = MacroData(
            jp_base_rate=jp_base_rate,
            jp_10y_yield=jp_10y_yield,
            usdjpy=usdjpy,
        )

        log.info(
            "macro_jp_fetch_complete",
            jp_base_rate=jp_base_rate,
            jp_10y_yield=jp_10y_yield,
            usdjpy=usdjpy,
        )
        return macro

    # -- Public API --------------------------------------------------------

    async def fetch_macro(self) -> MacroData:
        """Fetch Japan macroeconomic data with retry logic.

        Returns a :class:`MacroData` populated with ``jp_base_rate``,
        ``jp_10y_yield``, and ``usdjpy``. Fields that cannot be retrieved
        are left as ``None``.

        Raises:
            DataFetchError: If all retry attempts are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug("macro_jp_fetch_attempt", attempt=attempt)
                macro = await asyncio.to_thread(self._fetch_all_sync)
                log.info(
                    "macro_jp_fetch_success",
                    attempt=attempt,
                    has_base_rate=macro.jp_base_rate is not None,
                    has_10y=macro.jp_10y_yield is not None,
                    has_usdjpy=macro.usdjpy is not None,
                )
                return macro

            except RateLimitError:
                raise

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "macro_jp_fetch_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0 * attempt)

        msg = f"Japan macro data fetch failed after {self._max_retries} attempts"
        raise DataFetchError(
            msg,
            context={
                "series": _ALL_SERIES,
                "last_error": str(last_exc),
            },
        )
