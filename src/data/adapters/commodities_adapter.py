"""Commodities market data adapter using yfinance futures."""

from __future__ import annotations

from config.settings import Settings
from src.core.types import Market
from src.data.adapters.yfinance_base import YFinanceAdapterConfig, YFinanceBaseAdapter


class CommoditiesAdapter(YFinanceBaseAdapter):
    """Commodities market data adapter using yfinance futures contracts.

    Tracks major commodity futures: gold (GC=F), oil WTI (CL=F),
    silver (SI=F), natural gas (NG=F), and copper (HG=F).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            config=YFinanceAdapterConfig(
                market=Market.COMMODITIES,
                currency="USD",
                yaml_section="COMMODITIES",
                default_symbols=[
                    "GC=F",  # Gold futures
                    "CL=F",  # Crude Oil WTI futures
                    "SI=F",  # Silver futures
                    "NG=F",  # Natural Gas futures
                    "HG=F",  # Copper futures
                ],
                use_info=False,
                per_field=None,
                pbr_field=None,
                market_cap_field=None,
                static_extra={"contract_type": "futures"},
            ),
            settings=settings,
        )
