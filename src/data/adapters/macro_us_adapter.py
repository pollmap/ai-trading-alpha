"""US macroeconomic data adapter using FRED API."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from fredapi import Fred

from config.settings import Settings, get_settings
from src.core.constants import LLM_MAX_RETRIES
from src.core.exceptions import DataFetchError, RateLimitError
from src.core.logging import get_logger
from src.core.types import MacroData

log = get_logger(__name__)

# FRED series IDs used by this adapter.
_SERIES_FEDFUNDS = "FEDFUNDS"   # Effective Federal Funds Rate (monthly)
_SERIES_CPI = "CPIAUCSL"        # Consumer Price Index for All Urban Consumers (monthly)
_SERIES_VIX = "VIXCLS"          # CBOE Volatility Index (daily)
_SERIES_DGS10 = "DGS10"         # 10-Year Treasury Constant Maturity Rate (daily)
_SERIES_DGS2 = "DGS2"           # 2-Year Treasury Constant Maturity Rate (daily)

_ALL_SERIES: list[str] = [
    _SERIES_FEDFUNDS,
    _SERIES_CPI,
    _SERIES_VIX,
    _SERIES_DGS10,
    _SERIES_DGS2,
]

# Look-back window for fetching series data (enough to compute CPI YoY).
_LOOKBACK_DAYS = 400


class MacroUSAdapter:
    """Fetches US macroeconomic indicators from FRED.

    ``fredapi.Fred`` is synchronous, so all API calls are wrapped with
    ``asyncio.to_thread`` to avoid blocking the event loop.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        api_key: str = self._settings.fred_api_key.get_secret_value()
        if not api_key:
            msg = "fred_api_key is not configured in settings / .env"
            raise DataFetchError(msg)
        self._fred: Fred = Fred(api_key=api_key)
        self._max_retries: int = LLM_MAX_RETRIES
        log.info("macro_us_adapter_initialized")

    # ── Internal helpers ─────────────────────────────────────────

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
            # Drop NaN and take the most recent observation.
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

    def _compute_cpi_yoy_sync(self) -> float | None:
        """Calculate CPI Year-over-Year percentage change.

        ``YoY = (CPI_latest - CPI_12_months_ago) / CPI_12_months_ago * 100``
        """
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=_LOOKBACK_DAYS)
        try:
            series = self._fred.get_series(
                _SERIES_CPI,
                observation_start=start.strftime("%Y-%m-%d"),
                observation_end=end.strftime("%Y-%m-%d"),
            )
            if series is None or series.empty:
                log.warning("fred_cpi_series_empty")
                return None

            clean = series.dropna()
            if len(clean) < 13:
                log.warning(
                    "fred_cpi_insufficient_data",
                    available_points=len(clean),
                )
                return None

            latest_cpi = float(clean.iloc[-1])
            cpi_12m_ago = float(clean.iloc[-13])

            if cpi_12m_ago == 0:
                log.warning("fred_cpi_zero_denominator")
                return None

            yoy = (latest_cpi - cpi_12m_ago) / cpi_12m_ago * 100.0
            return round(yoy, 2)
        except Exception as exc:
            log.warning("fred_cpi_yoy_calculation_failed", error=str(exc))
            return None

    def _fetch_all_sync(self) -> MacroData:
        """Synchronous aggregate fetch — called via asyncio.to_thread."""
        fed_rate = self._fetch_series_sync(_SERIES_FEDFUNDS)
        vix = self._fetch_series_sync(_SERIES_VIX)
        dgs10 = self._fetch_series_sync(_SERIES_DGS10)
        dgs2 = self._fetch_series_sync(_SERIES_DGS2)
        cpi_yoy = self._compute_cpi_yoy_sync()

        macro = MacroData(
            us_fed_rate=fed_rate,
            us_cpi_yoy=cpi_yoy,
            vix=vix,
        )

        # Store treasury yields in a way accessible to callers who need them.
        # MacroData.extra is not available, but we log them for observability.
        log.info(
            "fred_fetch_complete",
            us_fed_rate=fed_rate,
            us_cpi_yoy=cpi_yoy,
            vix=vix,
            dgs10=dgs10,
            dgs2=dgs2,
            yield_spread=(
                round(dgs10 - dgs2, 3) if dgs10 is not None and dgs2 is not None else None
            ),
        )
        return macro

    # ── Public API ───────────────────────────────────────────────

    async def fetch_macro(self) -> MacroData:
        """Fetch US macroeconomic data from FRED with retry logic.

        Returns a :class:`MacroData` populated with ``us_fed_rate``,
        ``us_cpi_yoy``, and ``vix``.  Fields that cannot be retrieved are
        left as ``None``.

        Raises:
            DataFetchError: If all retry attempts are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                log.debug("macro_us_fetch_attempt", attempt=attempt)
                macro = await asyncio.to_thread(self._fetch_all_sync)
                log.info(
                    "macro_us_fetch_success",
                    attempt=attempt,
                    has_fed_rate=macro.us_fed_rate is not None,
                    has_cpi=macro.us_cpi_yoy is not None,
                    has_vix=macro.vix is not None,
                )
                return macro

            except RateLimitError:
                raise

            except Exception as exc:
                last_exc = exc
                log.warning(
                    "macro_us_fetch_retry",
                    attempt=attempt,
                    max_retries=self._max_retries,
                    error=str(exc),
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1.0 * attempt)

        msg = f"FRED macro data fetch failed after {self._max_retries} attempts"
        raise DataFetchError(
            msg,
            context={
                "series": _ALL_SERIES,
                "last_error": str(last_exc),
            },
        )
