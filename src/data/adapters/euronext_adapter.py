"""Europe (EURONEXT) market data adapter using yfinance."""

from __future__ import annotations

from config.settings import Settings
from src.core.types import Market
from src.data.adapters.yfinance_base import YFinanceAdapterConfig, YFinanceBaseAdapter


class EuronextAdapter(YFinanceBaseAdapter):
    """Europe (EURONEXT) stock market data adapter using yfinance for OHLCV data."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            config=YFinanceAdapterConfig(
                market=Market.EURONEXT,
                currency="EUR",
                yaml_section="EURONEXT",
                default_symbols=[
                    "MC.PA",    # LVMH (Paris)
                    "ASML.AS",  # ASML (Amsterdam)
                    "OR.PA",    # L'Oreal (Paris)
                    "SAN.PA",   # Sanofi (Paris)
                    "AIR.PA",   # Airbus (Paris)
                ],
                info_extra_fields={
                    "dividend_yield": "dividendYield",
                    "fifty_two_week_high": "fiftyTwoWeekHigh",
                    "fifty_two_week_low": "fiftyTwoWeekLow",
                },
            ),
            settings=settings,
        )
