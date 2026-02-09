"""China macroeconomic data adapter using yfinance and FRED."""

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

# FRED series IDs for China
_SERIES_CN_IR = "IRSTCI01CNM156N"  # China short-term interest rate (proxy for PBOC rate)
_SERIES_CN_10Y = "IRLTLT01CNM156N"  # China long-term government bond yield (10Y)

# yfinance ticker for USD/CNY exchange rate
_USDCNY_TICKER = "CNY=X"

_ALL_SERIES: list[str] = [_SERIES_CN_IR, _SERIES_CN_10Y]

_LOOKBACK_DAYS = 400


class MacroCNAdapter:
    """Fetches China macroeconomic indicators from FRED and yfinance.

    Retrieves:
    * PBOC base rate (proxy via short-term interest rate from FRED)
    * China 10-year government bond yield (from FRED)
    * USD/CNY exchange rate (from yfinance)

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
        log.info("macro_cn_adapter_initialized")

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

    def _fetch_usdcny_sync(self) -> float | None:
        """Fetch the latest USD/CNY exchange rate via yfinance."""
        try:
            ticker = yf.Ticker(_USDCNY_TICKER)
            hist = ticker.history(period="1d")
            if hist.empty:
                log.warning("yfinance_usdcny_empty")
                return None
            return float(hist.iloc[-1]["Close"])
        except Exception as exc:
            log.warning("yfinance_usdcny_fetch_failed", error=str(exc))
            return None

    def _fetch_all_sync(self) -> MacroData:
        """Synchronous aggregate fetch -- called via asyncio.to_thread."""
        cn_base_rate = self._fetch_series_sync(_SERIES_CN_IR)
        cn_10y_yield = self._fetch_series_sync(_SERIES_CN_10Y)
        usdcny = self._fetch_usdcny_sync()

        macro = MacroData(
            cn_base_rate=cn_base_rate,
            cn_10y_yield=cn_10y_yield,
            usdcny=usdcny,
        )

        log.info(
            "macro_cn_fetch_complete",
            cn_base_rate=cn_base_rate,
            cn_10y_yield=cn_10y_yield,
            usdcny=usdcny,
        )
        return macro

    # -- Public API --------------------------------------------------------

    async def fetch_macro(self) -> MacroData:
        """Fetch China macroeconomic data with retry logic.

        Returns a :class:`MacroData` populated with ``cn_base_rate``,
        ``cn_10y_yield``, and ``usdcny``. Fields that cannot be retrieved
        are left as ``None``.

        Raises:
            DataFetchError: If all retry attempts are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug("macro_cn_fetch_attempt", attempt=attempt)
                macro = await asyncio.to_thread(self._fetch_all_sync)
                log.info(
                    "macro_cn_fetch_success",
                    attempt=attempt,
                    has_base_rate=macro.cn_base_rate is not None,
                    has_10y=macro.cn_10y_yield is not None,
                    has_usdcny=macro.usdcny is not None,
                )
                return macro

            except RateLimitError:
                raise

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "macro_cn_fetch_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0 * attempt)

        msg = f"China macro data fetch failed after {self._max_retries} attempts"
        raise DataFetchError(
            msg,
            context={
                "series": _ALL_SERIES,
                "last_error": str(last_exc),
            },
        )
