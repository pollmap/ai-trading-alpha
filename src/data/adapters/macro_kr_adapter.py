"""Korean macroeconomic data adapter using ECOS (Bank of Korea) API."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.core.constants import LLM_MAX_RETRIES
from src.core.exceptions import DataFetchError
from src.core.logging import get_logger
from src.core.types import MacroData
from config.settings import get_settings

_ECOS_BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"

# ECOS stat codes (Bank of Korea statistical information system)
_STAT_BASE_RATE = "722Y001"       # Base interest rate
_ITEM_BASE_RATE = "0101000"       # Call rate target

_STAT_CPI = "901Y009"             # Consumer Price Index
_ITEM_CPI = "0"                   # CPI total (all items)

_STAT_EXCHANGE_RATE = "731Y001"   # Exchange rates
_ITEM_USDKRW = "0000001"         # USD/KRW

_FREQUENCY_MONTHLY = "M"
_FREQUENCY_DAILY = "D"

_MAX_RETRIES: int = LLM_MAX_RETRIES  # 3 attempts
_BACKOFF_BASE: float = 1.0
_HTTP_TIMEOUT: float = 15.0           # seconds

logger = get_logger(__name__)


class MacroKRAdapter:
    """Fetch Korean macroeconomic indicators from the Bank of Korea ECOS API.

    Retrieves:
    * Base interest rate (call rate target)
    * CPI year-over-year change
    * USD/KRW exchange rate

    All HTTP calls use ``httpx.AsyncClient`` so they are fully async.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key: str = settings.bok_api_key.get_secret_value()
        if not self._api_key:
            msg = "bok_api_key is not configured in settings"
            raise DataFetchError(msg)

    # ── public API ────────────────────────────────────────────────

    async def fetch_macro(self) -> MacroData:
        """Fetch the latest Korean macro indicators and return a ``MacroData``.

        Individual indicator failures are logged and the corresponding
        field is left as ``None`` so the pipeline continues with partial data.
        """
        now = datetime.now(timezone.utc)
        # Date range for queries
        end_str = now.strftime("%Y%m%d")
        # Go back ~4 months to ensure we capture the latest published value
        start_monthly_offset = _month_offset(now, -4).strftime("%Y%m")
        end_monthly = now.strftime("%Y%m")

        # Daily range for exchange rate (last 14 days for weekends/holidays)
        start_daily = (now - timedelta(days=14)).strftime("%Y%m%d")

        kr_base_rate = await self._fetch_base_rate(start_monthly_offset, end_monthly)
        kr_cpi_yoy = await self._fetch_cpi_yoy(start_monthly_offset, end_monthly)
        usdkrw = await self._fetch_usdkrw(start_daily, end_str)

        macro = MacroData(
            kr_base_rate=kr_base_rate,
            kr_cpi_yoy=kr_cpi_yoy,
            usdkrw=usdkrw,
        )

        logger.info(
            "macro_kr_fetch_complete",
            kr_base_rate=kr_base_rate,
            kr_cpi_yoy=kr_cpi_yoy,
            usdkrw=usdkrw,
        )
        return macro

    # ── private indicator fetchers ────────────────────────────────

    async def _fetch_base_rate(
        self,
        start: str,
        end: str,
    ) -> float | None:
        """Fetch the latest Bank of Korea base interest rate."""
        rows = await self._ecos_query(
            stat_code=_STAT_BASE_RATE,
            frequency=_FREQUENCY_MONTHLY,
            start=start,
            end=end,
            item_code=_ITEM_BASE_RATE,
            label="base_rate",
        )
        if not rows:
            return None
        return _extract_latest_value(rows)

    async def _fetch_cpi_yoy(
        self,
        start: str,
        end: str,
    ) -> float | None:
        """Fetch the latest Korean CPI year-over-year percentage change."""
        rows = await self._ecos_query(
            stat_code=_STAT_CPI,
            frequency=_FREQUENCY_MONTHLY,
            start=start,
            end=end,
            item_code=_ITEM_CPI,
            label="cpi_yoy",
        )
        if not rows:
            return None
        return _extract_latest_value(rows)

    async def _fetch_usdkrw(
        self,
        start: str,
        end: str,
    ) -> float | None:
        """Fetch the latest USD/KRW exchange rate (매매기준율)."""
        rows = await self._ecos_query(
            stat_code=_STAT_EXCHANGE_RATE,
            frequency=_FREQUENCY_DAILY,
            start=start,
            end=end,
            item_code=_ITEM_USDKRW,
            label="usdkrw",
        )
        if not rows:
            return None
        return _extract_latest_value(rows)

    # ── ECOS HTTP layer ───────────────────────────────────────────

    async def _ecos_query(
        self,
        *,
        stat_code: str,
        frequency: str,
        start: str,
        end: str,
        item_code: str,
        label: str,
    ) -> list[dict[str, Any]]:
        """Issue a single ECOS StatisticSearch request with retry.

        URL pattern::

            /StatisticSearch/{api_key}/json/kr/1/100/{stat_code}/{freq}/{start}/{end}/{item_code}

        Returns the ``row`` list from the JSON response, or an empty list on
        failure.
        """
        url = (
            f"{_ECOS_BASE_URL}/{self._api_key}/json/kr/1/100"
            f"/{stat_code}/{frequency}/{start}/{end}/{item_code}"
        )

        last_exc: BaseException | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                    response = await client.get(url)
                    response.raise_for_status()

                body: dict[str, Any] = response.json()

                # ECOS wraps data in {"StatisticSearch": {"row": [...]}}
                search_result = body.get("StatisticSearch")
                if search_result is None:
                    # ECOS returns error info at the top level on failure
                    error_result = body.get("RESULT")
                    error_msg = (
                        error_result.get("MESSAGE", "unknown ECOS error")
                        if error_result
                        else f"unexpected ECOS response structure for {label}"
                    )
                    logger.warning(
                        "ecos_api_error",
                        label=label,
                        stat_code=stat_code,
                        error=error_msg,
                    )
                    return []

                rows: list[dict[str, Any]] = search_result.get("row", [])
                logger.debug(
                    "ecos_query_ok",
                    label=label,
                    stat_code=stat_code,
                    row_count=len(rows),
                )
                return rows

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                logger.warning(
                    "ecos_http_error",
                    label=label,
                    attempt=attempt,
                    status_code=exc.response.status_code,
                    error=str(exc),
                )
            except httpx.RequestError as exc:
                last_exc = exc
                logger.warning(
                    "ecos_request_error",
                    label=label,
                    attempt=attempt,
                    error=str(exc),
                )

            if attempt < _MAX_RETRIES:
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info(
                    "ecos_retry_wait",
                    label=label,
                    wait_seconds=wait,
                    attempt=attempt,
                )
                await asyncio.sleep(wait)

        logger.error(
            "ecos_query_failed",
            label=label,
            stat_code=stat_code,
            error=str(last_exc),
        )
        return []


# ── module-level helpers ──────────────────────────────────────────


def _extract_latest_value(rows: list[dict[str, Any]]) -> float | None:
    """Return the numeric value from the last (most recent) row.

    ECOS rows carry the value in the ``DATA_VALUE`` field as a string.
    """
    if not rows:
        return None
    try:
        raw = rows[-1].get("DATA_VALUE", "")
        # ECOS sometimes uses commas as thousands separators
        cleaned = str(raw).replace(",", "").strip()
        return float(cleaned)
    except (ValueError, TypeError) as exc:
        logger.warning("ecos_parse_error", raw_value=rows[-1], error=str(exc))
        return None


def _month_offset(dt: datetime, months: int) -> datetime:
    """Return *dt* shifted by *months* (negative = past).

    Simple arithmetic that handles year boundaries.
    """
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, 28)  # safe for all months
    return dt.replace(year=year, month=month, day=day)
