"""Global bond market data adapter using yfinance bond ETFs."""

from __future__ import annotations

from config.settings import Settings
from src.core.types import Market
from src.data.adapters.yfinance_base import YFinanceAdapterConfig, YFinanceBaseAdapter


class BondAdapter(YFinanceBaseAdapter):
    """Global bond market data adapter using yfinance bond ETFs.

    Tracks major bond ETFs (TLT, IEF, LQD, HYG, EMB) as proxies for
    different bond market segments.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(
            config=YFinanceAdapterConfig(
                market=Market.BOND,
                currency="USD",
                yaml_section="BOND",
                default_symbols=[
                    "TLT",  # iShares 20+ Year Treasury Bond ETF
                    "IEF",  # iShares 7-10 Year Treasury Bond ETF
                    "LQD",  # iShares iBoxx Investment Grade Corporate Bond ETF
                    "HYG",  # iShares iBoxx High Yield Corporate Bond ETF
                    "EMB",  # iShares J.P. Morgan USD Emerging Markets Bond ETF
                ],
                per_field=None,
                pbr_field=None,
                market_cap_field="totalAssets",
                info_extra_fields={
                    "yield": "yield",
                    "ytd_return": "ytdReturn",
                    "three_year_avg_return": "threeYearAverageReturn",
                    "expense_ratio": "annualReportExpenseRatio",
                },
            ),
            settings=settings,
        )
