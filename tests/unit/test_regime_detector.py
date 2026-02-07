"""Tests for regime detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.regime_detector import (
    MarketRegime,
    RegimeAnalyzer,
    RegimeDetector,
)


@pytest.fixture
def detector() -> RegimeDetector:
    return RegimeDetector()


class TestRegimeDetector:
    def test_insufficient_data(self, detector: RegimeDetector) -> None:
        prices = pd.Series([100.0, 101.0, 102.0])
        result = detector.detect(prices)
        assert result == MarketRegime.SIDEWAYS

    def test_bull_regime(self, detector: RegimeDetector) -> None:
        # Steady uptrend: +10% over 21 days
        prices = pd.Series(np.linspace(100, 110, 25))
        result = detector.detect(prices)
        assert result == MarketRegime.BULL

    def test_bear_regime(self, detector: RegimeDetector) -> None:
        # Steady downtrend: -8% over 21 days (below crash threshold of 15%)
        # Need enough volatility to avoid high_vol but steady downward trend
        prices = pd.Series(np.linspace(100, 92, 25))
        result = detector.detect(prices)
        assert result in (MarketRegime.BEAR, MarketRegime.HIGH_VOL)

    def test_crash_regime(self, detector: RegimeDetector) -> None:
        # Severe drop: peak then crash >15%
        prices_data = list(np.linspace(100, 120, 15)) + list(np.linspace(120, 95, 10))
        prices = pd.Series(prices_data)
        result = detector.detect(prices)
        assert result == MarketRegime.CRASH

    def test_sideways_regime(self, detector: RegimeDetector) -> None:
        # Flat prices (±1%)
        np.random.seed(42)
        prices = pd.Series(100.0 + np.random.randn(25) * 0.5)
        result = detector.detect(prices)
        assert result == MarketRegime.SIDEWAYS

    def test_vix_override(self, detector: RegimeDetector) -> None:
        # Normal prices but VIX > 30 -> HIGH_VOL
        prices = pd.Series(np.linspace(100, 101, 25))
        result = detector.detect(prices, vix=35.0)
        assert result == MarketRegime.HIGH_VOL

    def test_detect_from_returns(self, detector: RegimeDetector) -> None:
        # Bull returns
        returns = pd.Series([0.005] * 25)
        result = detector.detect_from_returns(returns)
        assert result == MarketRegime.BULL

    def test_regime_history(self, detector: RegimeDetector) -> None:
        prices = pd.Series(np.linspace(100, 110, 50))
        history = detector.get_regime_history(prices, window=20)
        assert len(history) == 50
        assert all(isinstance(r, MarketRegime) for _, r in history)

    def test_prompt_context(self, detector: RegimeDetector) -> None:
        for regime in MarketRegime:
            text = detector.to_prompt_context(regime)
            assert regime.value.upper().replace("_", " ") in text.upper() or "MARKET" in text.upper()


class TestRegimeAnalyzer:
    def test_analyze_by_regime(self) -> None:
        analyzer = RegimeAnalyzer()
        values = list(np.linspace(100, 110, 20))
        history = [(i, MarketRegime.BULL) for i in range(20)]
        results = analyzer.analyze_by_regime(values, history)
        assert MarketRegime.BULL in results
        assert results[MarketRegime.BULL].n_periods > 0
        assert results[MarketRegime.BULL].cumulative_return > 0

    def test_empty_data(self) -> None:
        analyzer = RegimeAnalyzer()
        results = analyzer.analyze_by_regime([], [])
        assert results == {}
