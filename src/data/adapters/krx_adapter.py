"""KRX market data adapter using pykrx library."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from pykrx import stock as pykrx_stock

from src.core.constants import CURRENCY_KRW, LLM_MAX_RETRIES
from src.core.exceptions import DataFetchError
from src.core.interfaces import BaseMarketDataAdapter
from src.core.logging import get_logger
from src.core.types import Market, SymbolData

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_MARKETS_YAML = _PROJECT_ROOT / "config" / "markets.yaml"

_MAX_RETRIES: int = LLM_MAX_RETRIES  # 3 attempts
_BACKOFF_BASE: float = 1.0           # 1s, 2s, 4s exponential backoff
_SEMAPHORE_LIMIT: int = 5            # max concurrent pykrx calls per second

logger = get_logger(__name__)


def _load_krx_symbols() -> list[str]:
    """Load KRX symbol list from markets.yaml."""
    with _MARKETS_YAML.open(encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)
    symbols: list[str] = config.get("KRX", {}).get("symbols", [])
    if not symbols:
        msg = "No KRX symbols found in markets.yaml"
        raise DataFetchError(msg, context={"path": str(_MARKETS_YAML)})
    return [str(s) for s in symbols]


class KRXAdapter(BaseMarketDataAdapter):
    """Fetch KRX market data (OHLCV, fundamentals, investor trading) via pykrx.

    All pykrx calls are synchronous and therefore run inside
    ``asyncio.to_thread()`` to avoid blocking the event loop.
    Concurrent calls are throttled with an ``asyncio.Semaphore``.
    """

    def __init__(self, symbols: list[str] | None = None) -> None:
        self._symbols: list[str] = symbols or _load_krx_symbols()
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(_SEMAPHORE_LIMIT)

    # ── public API ────────────────────────────────────────────────

    async def fetch_latest(self) -> dict[str, SymbolData]:
        """Fetch the latest OHLCV + fundamental data for every configured symbol.

        Returns a mapping of ``{symbol: SymbolData}``.  Symbols that fail
        after all retries are logged and excluded from the result rather than
        raising, so the pipeline can continue with a partial snapshot.
        """
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        # pykrx uses KST dates internally; fetch today and yesterday to
        # ensure we get at least one trading day of data.
        yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime(
            "%Y%m%d",
        )

        tasks = [
            self._fetch_symbol(symbol, yesterday_str, today_str)
            for symbol in self._symbols
        ]
        results: list[SymbolData | None] = await asyncio.gather(*tasks)

        data: dict[str, SymbolData] = {}
        for sd in results:
            if sd is not None:
                data[sd.symbol] = sd

        logger.info(
            "krx_fetch_latest_complete",
            fetched=len(data),
            total=len(self._symbols),
            missing=[s for s in self._symbols if s not in data],
        )
        return data

    async def fetch_investor_trading(
        self,
        date_str: str,
    ) -> dict[str, dict[str, Any]]:
        """Fetch investor-type buy/sell data for each configured symbol.

        Args:
            date_str: Date in ``YYYYMMDD`` format.

        Returns:
            ``{symbol: {"institution": ..., "foreign": ..., "individual": ...}}``
            where each value contains net buy volume for the day.
        """
        tasks = [
            self._fetch_investor_for_symbol(symbol, date_str)
            for symbol in self._symbols
        ]
        results: list[tuple[str, dict[str, Any]] | None] = await asyncio.gather(*tasks)

        investor_data: dict[str, dict[str, Any]] = {}
        for item in results:
            if item is not None:
                symbol, info = item
                investor_data[symbol] = info

        logger.info(
            "krx_investor_trading_complete",
            date=date_str,
            fetched=len(investor_data),
            total=len(self._symbols),
        )
        return investor_data

    # ── private helpers ───────────────────────────────────────────

    async def _fetch_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> SymbolData | None:
        """Fetch OHLCV + fundamentals for a single symbol with retry."""
        ohlcv = await self._call_with_retry(
            pykrx_stock.get_market_ohlcv_by_date,
            start_date,
            end_date,
            symbol,
            context_symbol=symbol,
            context_call="get_market_ohlcv_by_date",
        )
        if ohlcv is None or ohlcv.empty:
            logger.warning("krx_ohlcv_empty", symbol=symbol)
            return None

        # Take the most recent row
        latest = ohlcv.iloc[-1]

        # Fundamentals (PER, PBR, market cap)
        fundamentals = await self._call_with_retry(
            pykrx_stock.get_market_fundamental_by_date,
            start_date,
            end_date,
            symbol,
            context_symbol=symbol,
            context_call="get_market_fundamental_by_date",
        )
        per: float | None = None
        pbr: float | None = None
        if fundamentals is not None and not fundamentals.empty:
            fund_latest = fundamentals.iloc[-1]
            raw_per = fund_latest.get("PER", None)
            raw_pbr = fund_latest.get("PBR", None)
            per = float(raw_per) if raw_per is not None and raw_per != 0 else None
            pbr = float(raw_pbr) if raw_pbr is not None and raw_pbr != 0 else None

        cap_df = await self._call_with_retry(
            pykrx_stock.get_market_cap_by_date,
            start_date,
            end_date,
            symbol,
            context_symbol=symbol,
            context_call="get_market_cap_by_date",
        )
        market_cap: float | None = None
        if cap_df is not None and not cap_df.empty:
            raw_cap = cap_df.iloc[-1].get("시가총액", None)
            market_cap = float(raw_cap) if raw_cap is not None else None

        return SymbolData(
            symbol=symbol,
            market=Market.KRX,
            open=float(latest.get("시가", 0)),
            high=float(latest.get("고가", 0)),
            low=float(latest.get("저가", 0)),
            close=float(latest.get("종가", 0)),
            volume=float(latest.get("거래량", 0)),
            currency=CURRENCY_KRW,
            per=per,
            pbr=pbr,
            market_cap=market_cap,
        )

    async def _fetch_investor_for_symbol(
        self,
        symbol: str,
        date_str: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """Fetch investor trading breakdown for one symbol."""
        df = await self._call_with_retry(
            pykrx_stock.get_market_trading_value_by_date,
            date_str,
            date_str,
            symbol,
            context_symbol=symbol,
            context_call="get_market_trading_value_by_date",
        )
        if df is None or df.empty:
            logger.warning("krx_investor_empty", symbol=symbol, date=date_str)
            return None

        row = df.iloc[-1]
        return (
            symbol,
            {
                "institution": float(row.get("기관합계", 0)),
                "foreign": float(row.get("외국인합계", 0)),
                "individual": float(row.get("개인", 0)),
            },
        )

    async def _call_with_retry(
        self,
        func: Any,
        *args: Any,
        context_symbol: str = "",
        context_call: str = "",
    ) -> Any:
        """Execute a synchronous pykrx function inside ``asyncio.to_thread()``
        with semaphore-based rate limiting and exponential-backoff retry.

        Returns ``None`` if all retry attempts are exhausted.
        """
        last_exc: BaseException | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with self._semaphore:
                    result: Any = await asyncio.to_thread(func, *args)
                return result
            except Exception as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "krx_call_retry",
                    symbol=context_symbol,
                    call=context_call,
                    attempt=attempt,
                    max_retries=_MAX_RETRIES,
                    wait_seconds=wait,
                    error=str(exc),
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(wait)

        logger.error(
            "krx_call_failed",
            symbol=context_symbol,
            call=context_call,
            error=str(last_exc),
        )
        return None
