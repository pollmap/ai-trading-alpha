"""London (LSE / London Stock Exchange) market data adapter using yfinance."""

from __future__ import annotations

from config.settings import Settings
from src.core.types import Market
from src.data.adapters.yfinance_base import YFinanceAdapterConfig, YFinanceBaseAdapter


class LSEAdapter(YFinanceBaseAdapter):
    """London (LSE) stock market data adapter using yfinance for OHLCV data."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            config=YFinanceAdapterConfig(
                market=Market.LSE,
                currency="GBP",
                yaml_section="LSE",
                default_symbols=[
                    "SHEL.L",  # Shell
                    "AZN.L",   # AstraZeneca
                    "HSBA.L",  # HSBC
                    "ULVR.L",  # Unilever
                    "BP.L",    # BP
                ],
                info_extra_fields={
                    "dividend_yield": "dividendYield",
                    "fifty_two_week_high": "fiftyTwoWeekHigh",
                    "fifty_two_week_low": "fiftyTwoWeekLow",
                },
            ),
            settings=settings,
        )
