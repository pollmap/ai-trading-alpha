"""Tests for technical indicators engine."""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.data.indicators import IndicatorEngine, TechnicalIndicators


@pytest.fixture
def engine() -> IndicatorEngine:
    return IndicatorEngine()


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate 250 days of synthetic OHLCV data."""
    np.random.seed(42)
    n = 250
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 10.0)  # keep positive
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.005),
        "high": close * (1 + abs(np.random.randn(n) * 0.01)),
        "low": close * (1 - abs(np.random.randn(n) * 0.01)),
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, size=n).astype(float),
    })


class TestIndicatorEngine:
    def test_empty_dataframe(self, engine: IndicatorEngine) -> None:
        result = engine.calculate("TEST", pd.DataFrame())
        assert result.symbol == "TEST"
        assert result.rsi_14 is None

    def test_missing_columns(self, engine: IndicatorEngine) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        result = engine.calculate("TEST", df)
        assert result.rsi_14 is None

    def test_full_calculation(self, engine: IndicatorEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.calculate("AAPL", sample_ohlcv)
        assert result.symbol == "AAPL"
        assert result.rsi_14 is not None
        assert 0.0 <= result.rsi_14 <= 100.0
        assert result.sma_20 is not None
        assert result.sma_50 is not None
        assert result.sma_200 is not None
        assert result.macd_line is not None
        assert result.macd_signal is not None
        assert result.macd_histogram is not None
        assert result.bb_upper is not None
        assert result.bb_lower is not None
        assert result.bb_width is not None
        assert result.atr_14 is not None
        assert result.atr_14 > 0
        assert result.volume_ratio is not None
        assert result.historical_volatility_20 is not None

    def test_insufficient_data_for_sma200(self, engine: IndicatorEngine) -> None:
        np.random.seed(42)
        n = 100  # Not enough for SMA200
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10.0)
        df = pd.DataFrame({
            "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close,
            "volume": np.full(n, 1_000_000.0),
        })
        result = engine.calculate("TEST", df)
        assert result.sma_200 is None
        assert result.sma_50 is not None
        assert result.rsi_14 is not None

    def test_batch_calculation(self, engine: IndicatorEngine, sample_ohlcv: pd.DataFrame) -> None:
        batch = {"AAPL": sample_ohlcv, "GOOG": sample_ohlcv}
        results = engine.calculate_batch(batch)
        assert len(results) == 2
        assert "AAPL" in results
        assert "GOOG" in results

    def test_prompt_line(self, engine: IndicatorEngine, sample_ohlcv: pd.DataFrame) -> None:
        result = engine.calculate("BTCUSDT", sample_ohlcv)
        line = result.to_prompt_line()
        assert "BTCUSDT:" in line
        assert "RSI=" in line
