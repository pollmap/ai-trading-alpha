"""Tests for global market expansion — types, constants, adapters."""

from __future__ import annotations

import pytest

from src.core.types import Market, MacroData, MarketSnapshot, SymbolData
from src.core import constants
from datetime import datetime, timezone


class TestMarketEnum:
    """Tests for expanded Market enum."""

    def test_original_markets_exist(self) -> None:
        assert Market.KRX.value == "KRX"
        assert Market.US.value == "US"
        assert Market.CRYPTO.value == "CRYPTO"

    def test_new_markets_exist(self) -> None:
        assert Market.JPX.value == "JPX"
        assert Market.SSE.value == "SSE"
        assert Market.HKEX.value == "HKEX"
        assert Market.EURONEXT.value == "EURONEXT"
        assert Market.LSE.value == "LSE"
        assert Market.BOND.value == "BOND"
        assert Market.COMMODITIES.value == "COMMODITIES"

    def test_total_market_count(self) -> None:
        assert len(Market) == 10


class TestConstants:
    """Tests for new market constants."""

    def test_new_market_codes(self) -> None:
        assert constants.MARKET_JPX == "JPX"
        assert constants.MARKET_SSE == "SSE"
        assert constants.MARKET_HKEX == "HKEX"
        assert constants.MARKET_EURONEXT == "EURONEXT"
        assert constants.MARKET_LSE == "LSE"
        assert constants.MARKET_BOND == "BOND"
        assert constants.MARKET_COMMODITIES == "COMMODITIES"

    def test_new_currencies(self) -> None:
        assert constants.CURRENCY_JPY == "JPY"
        assert constants.CURRENCY_CNY == "CNY"
        assert constants.CURRENCY_HKD == "HKD"
        assert constants.CURRENCY_EUR == "EUR"
        assert constants.CURRENCY_GBP == "GBP"

    def test_cache_ttls(self) -> None:
        assert constants.CACHE_TTL_JPX == 60
        assert constants.CACHE_TTL_SSE == 60
        assert constants.CACHE_TTL_BOND == 300  # slower
        assert constants.CACHE_TTL_COMMODITIES == 30


class TestMacroData:
    """Tests for expanded MacroData fields."""

    def test_new_macro_fields(self) -> None:
        macro = MacroData(
            jp_base_rate=0.10,
            cn_base_rate=3.45,
            ecb_rate=4.00,
            boe_rate=5.25,
            usdjpy=148.50,
            usdcny=7.15,
            eurusd=1.085,
            gbpusd=1.265,
            usdhkd=7.82,
            gold_price=2050.0,
            oil_wti_price=75.50,
            us_10y_yield=4.25,
            jp_10y_yield=0.85,
            de_10y_yield=2.35,
            cn_10y_yield=2.70,
        )
        assert macro.jp_base_rate == 0.10
        assert macro.gold_price == 2050.0
        assert macro.de_10y_yield == 2.35

    def test_backward_compatible(self) -> None:
        """Old code creating MacroData with only original fields should still work."""
        macro = MacroData(
            kr_base_rate=3.50,
            us_fed_rate=5.25,
            usdkrw=1320.0,
            vix=18.5,
        )
        assert macro.jp_base_rate is None
        assert macro.gold_price is None
        assert macro.us_10y_yield is None


class TestMarketSnapshotPrompt:
    """Test that to_prompt_summary includes new macro fields."""

    def test_includes_global_macro(self) -> None:
        snapshot = MarketSnapshot(
            snapshot_id="snap-001",
            timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
            market=Market.JPX,
            symbols={
                "7203.T": SymbolData(
                    symbol="7203.T",
                    market=Market.JPX,
                    open=2500.0,
                    high=2550.0,
                    low=2480.0,
                    close=2530.0,
                    volume=10_000_000,
                    currency="JPY",
                ),
            },
            macro=MacroData(
                jp_base_rate=0.10,
                usdjpy=148.50,
                gold_price=2050.0,
            ),
        )
        summary = snapshot.to_prompt_summary()
        assert "JP Base Rate=0.10%" in summary
        assert "USD/JPY=148.50" in summary
        assert "Gold=2050.00" in summary
        assert "7203.T" in summary
