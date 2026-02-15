"""China (SSE / Shanghai Stock Exchange) market data adapter using yfinance."""

from __future__ import annotations

from config.settings import Settings
from src.core.types import Market
from src.data.adapters.yfinance_base import YFinanceAdapterConfig, YFinanceBaseAdapter


class SSEAdapter(YFinanceBaseAdapter):
    """China (SSE) stock market data adapter using yfinance for OHLCV data."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            config=YFinanceAdapterConfig(
                market=Market.SSE,
                currency="CNY",
                yaml_section="SSE",
                default_symbols=[
                    "600519.SS",  # Kweichow Moutai
                    "601318.SS",  # Ping An Insurance
                    "600036.SS",  # China Merchants Bank
                    "000858.SZ",  # Wuliangye Yibin
                    "601899.SS",  # Zijin Mining
                ],
                info_extra_fields={
                    "dividend_yield": "dividendYield",
                    "fifty_two_week_high": "fiftyTwoWeekHigh",
                    "fifty_two_week_low": "fiftyTwoWeekLow",
                },
            ),
            settings=settings,
        )
