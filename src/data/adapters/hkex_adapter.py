"""Hong Kong (HKEX) market data adapter using yfinance."""

from __future__ import annotations

from config.settings import Settings
from src.core.types import Market
from src.data.adapters.yfinance_base import YFinanceAdapterConfig, YFinanceBaseAdapter


class HKEXAdapter(YFinanceBaseAdapter):
    """Hong Kong (HKEX) stock market data adapter using yfinance for OHLCV data."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            config=YFinanceAdapterConfig(
                market=Market.HKEX,
                currency="HKD",
                yaml_section="HKEX",
                default_symbols=[
                    "0700.HK",  # Tencent
                    "9988.HK",  # Alibaba
                    "0941.HK",  # China Mobile
                    "1299.HK",  # AIA Group
                    "2318.HK",  # Ping An Insurance
                ],
                info_extra_fields={
                    "dividend_yield": "dividendYield",
                    "fifty_two_week_high": "fiftyTwoWeekHigh",
                    "fifty_two_week_low": "fiftyTwoWeekLow",
                },
            ),
            settings=settings,
        )
