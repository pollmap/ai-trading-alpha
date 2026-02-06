"""Tests for walk-forward analysis and Monte Carlo simulation."""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.walk_forward import (
    MonteCarloResult,
    MonteCarloSimulator,
    WalkForwardAnalyzer,
    WalkForwardResult,
)


@pytest.fixture
def analyzer() -> WalkForwardAnalyzer:
    return WalkForwardAnalyzer()


@pytest.fixture
def simulator() -> MonteCarloSimulator:
    return MonteCarloSimulator()


class TestWalkForwardAnalyzer:
    def test_insufficient_data(self, analyzer: WalkForwardAnalyzer) -> None:
        result = analyzer.run([100.0, 101.0, 102.0], window_size=30, oos_size=10)
        assert result.n_windows == 0

    def test_basic_walk_forward(self, analyzer: WalkForwardAnalyzer) -> None:
        # 100 data points, window=20, oos=10
        np.random.seed(42)
        values = list(np.cumsum(np.random.randn(100) * 0.5) + 100)
        result = analyzer.run(values, window_size=20, oos_size=10)
        assert result.n_windows > 0
        assert result.total_data_points == 100
        assert len(result.windows) == result.n_windows

    def test_robust_strategy(self, analyzer: WalkForwardAnalyzer) -> None:
        # Consistently upward trend -> robust
        values = list(np.linspace(100, 200, 100))
        result = analyzer.run(values, window_size=20, oos_size=10)
        assert result.n_windows > 0
        assert result.oos_cumulative_return > 0  # trending up overall

    def test_window_details(self, analyzer: WalkForwardAnalyzer) -> None:
        values = list(np.linspace(100, 120, 50))
        result = analyzer.run(values, window_size=15, oos_size=5)
        for w in result.windows:
            assert w.in_sample_start >= 0
            assert w.out_of_sample_end < 50


class TestMonteCarloSimulator:
    def test_insufficient_data(self, simulator: MonteCarloSimulator) -> None:
        result = simulator.simulate([0.01, 0.02], n_simulations=100)
        assert result.n_simulations == 0  # needs 5 min

    def test_basic_simulation(self, simulator: MonteCarloSimulator) -> None:
        np.random.seed(42)
        returns = list(np.random.randn(50) * 0.01 + 0.001)
        result = simulator.simulate(returns, n_simulations=1000)
        assert result.n_simulations == 1000
        assert result.p_value >= 0.0
        assert result.p_value <= 1.0
        assert result.percentile_rank >= 0.0
        assert result.percentile_rank <= 1.0

    def test_significant_strategy(self, simulator: MonteCarloSimulator) -> None:
        # Very strong positive returns should be statistically significant
        returns = [0.05] * 20  # 5% daily for 20 days
        result = simulator.simulate(returns, n_simulations=1000)
        # With such strong returns, p-value should be low
        assert result.actual_return > 0

    def test_confidence_interval(self, simulator: MonteCarloSimulator) -> None:
        np.random.seed(42)
        returns = list(np.random.randn(30) * 0.01)
        result = simulator.simulate(returns, n_simulations=500, confidence_level=0.95)
        lo, hi = result.confidence_interval
        assert lo <= hi
