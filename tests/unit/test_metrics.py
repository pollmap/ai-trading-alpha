"""Tests for analytics metrics engine."""

from __future__ import annotations

import pytest

from src.analytics.metrics import MetricsEngine, PerformanceMetrics


@pytest.fixture
def engine() -> MetricsEngine:
    return MetricsEngine()


class TestCalculateMetrics:
    def test_empty_values(self, engine: MetricsEngine) -> None:
        result = engine.calculate([], [])
        assert result.cumulative_return == 0.0
        assert result.sharpe_ratio == 0.0

    def test_single_value(self, engine: MetricsEngine) -> None:
        result = engine.calculate([100.0], ["2025-01-01"])
        assert result.cumulative_return == 0.0

    def test_positive_return(self, engine: MetricsEngine) -> None:
        values = [100.0, 105.0, 110.0, 115.0, 121.0]
        timestamps = [f"2025-01-0{i}" for i in range(1, 6)]
        result = engine.calculate(values, timestamps)
        assert result.cumulative_return > 0
        assert abs(result.cumulative_return - 0.21) < 0.01

    def test_negative_return(self, engine: MetricsEngine) -> None:
        values = [100.0, 95.0, 90.0]
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03"]
        result = engine.calculate(values, timestamps)
        assert result.cumulative_return < 0


class TestSharpeRatio:
    def test_positive_sharpe_consistent_gains(self, engine: MetricsEngine) -> None:
        # Steadily increasing values -> positive Sharpe
        values = [100.0 + i * 1.0 for i in range(20)]
        timestamps = [f"2025-01-{i+1:02d}" for i in range(20)]
        result = engine.calculate(values, timestamps)
        assert result.sharpe_ratio > 0

    def test_volatile_series_lower_sharpe(self, engine: MetricsEngine) -> None:
        values = [100.0, 110.0, 95.0, 115.0, 90.0, 120.0, 85.0, 110.0]
        timestamps = [f"2025-01-{i+1:02d}" for i in range(8)]
        result = engine.calculate(values, timestamps)
        assert abs(result.sharpe_ratio) < 10


class TestMaxDrawdown:
    def test_drawdown_from_peak(self, engine: MetricsEngine) -> None:
        values = [100.0, 120.0, 90.0, 110.0]
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        result = engine.calculate(values, timestamps)
        # Peak 120, trough 90 -> -25%
        assert abs(result.max_drawdown_pct - (-0.25)) < 0.01

    def test_no_drawdown(self, engine: MetricsEngine) -> None:
        values = [100.0, 110.0, 120.0, 130.0]
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        result = engine.calculate(values, timestamps)
        assert result.max_drawdown_pct == 0.0


class TestVolatility:
    def test_volatile_series_positive(self, engine: MetricsEngine) -> None:
        values = [100.0, 110.0, 90.0, 105.0, 95.0]
        timestamps = [f"2025-01-{i+1:02d}" for i in range(5)]
        result = engine.calculate(values, timestamps)
        assert result.volatility > 0

    def test_constant_series_zero_vol(self, engine: MetricsEngine) -> None:
        values = [100.0, 100.0, 100.0, 100.0]
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]
        result = engine.calculate(values, timestamps)
        assert result.volatility == 0.0


class TestTradingMetrics:
    def test_with_trades(self, engine: MetricsEngine) -> None:
        values = [100.0, 110.0, 115.0]
        timestamps = ["2025-01-01", "2025-01-02", "2025-01-03"]
        trades = [
            {"realized_pnl": 5.0},
            {"realized_pnl": -2.0},
            {"realized_pnl": 8.0},
        ]
        result = engine.calculate(values, timestamps, trades=trades)
        assert result.trade_count == 3
        assert result.win_rate > 0.5
        assert result.profit_factor > 1.0

    def test_no_trades(self, engine: MetricsEngine) -> None:
        values = [100.0, 110.0]
        timestamps = ["2025-01-01", "2025-01-02"]
        result = engine.calculate(values, timestamps)
        assert result.trade_count == 0
        assert result.win_rate == 0.0


class TestCostAdjustedAlpha:
    def test_caa_positive(self, engine: MetricsEngine) -> None:
        values = [100.0, 120.0]  # 20% return
        timestamps = ["2025-01-01", "2025-01-02"]
        result = engine.calculate(
            values, timestamps,
            total_api_cost=1.0,
            benchmark_return=0.10,
        )
        # CAA = (0.20 - 0.10) / 1.0 = 0.10
        assert result.cost_adjusted_alpha > 0

    def test_caa_zero_cost(self, engine: MetricsEngine) -> None:
        values = [100.0, 120.0]
        timestamps = ["2025-01-01", "2025-01-02"]
        result = engine.calculate(values, timestamps, total_api_cost=0.0)
        assert result.cost_adjusted_alpha == 0.0


class TestPerformanceMetricsDataclass:
    def test_all_fields_present(self) -> None:
        metrics = PerformanceMetrics(
            cumulative_return=0.15,
            annualized_return=0.45,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-5000.0,
            max_drawdown_pct=-0.05,
            calmar_ratio=9.0,
            volatility=0.15,
            win_rate=0.6,
            profit_factor=2.0,
            trade_count=50,
            avg_holding_period_days=3.0,
            cost_per_signal=0.05,
            cost_adjusted_alpha=0.10,
            signal_accuracy=0.55,
        )
        assert metrics.cumulative_return == 0.15
        assert metrics.trade_count == 50
