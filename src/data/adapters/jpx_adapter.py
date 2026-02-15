"""Japan (JPX / Tokyo Stock Exchange) market data adapter using yfinance."""

from __future__ import annotations

from config.settings import Settings
from src.core.types import Market
from src.data.adapters.yfinance_base import YFinanceAdapterConfig, YFinanceBaseAdapter


class JPXAdapter(YFinanceBaseAdapter):
    """Japan (JPX) stock market data adapter using yfinance for OHLCV data."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            config=YFinanceAdapterConfig(
                market=Market.JPX,
                currency="JPY",
                yaml_section="JPX",
                default_symbols=[
                    "7203.T",   # Toyota
                    "6758.T",   # Sony
                    "9984.T",   # SoftBank
                    "6861.T",   # Keyence
                    "8306.T",   # MUFG
                ],
                info_extra_fields={
                    "dividend_yield": "dividendYield",
                    "fifty_two_week_high": "fiftyTwoWeekHigh",
                    "fifty_two_week_low": "fiftyTwoWeekLow",
                },
            ),
            settings=settings,
        )
