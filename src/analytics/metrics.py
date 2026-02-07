"""Performance metrics engine."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging import get_logger

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """Calculated performance metrics for a portfolio."""

    cumulative_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    calmar_ratio: float
    volatility: float
    win_rate: float
    profit_factor: float
    trade_count: int
    avg_holding_period_days: float
    cost_per_signal: float
    cost_adjusted_alpha: float
    signal_accuracy: float


class MetricsEngine:
    """Calculate performance metrics from portfolio time-series.

    All calculations use daily returns (trading-day basis).
    """

    def calculate(
        self,
        portfolio_values: list[float],
        timestamps: list[str],
        trades: list[dict] | None = None,
        total_api_cost: float = 0.0,
        benchmark_return: float = 0.0,
        risk_free_rate: float = 0.0,
    ) -> PerformanceMetrics:
        """Calculate all performance metrics.

        Args:
            portfolio_values: Time-series of total portfolio values.
            timestamps: Corresponding timestamps.
            trades: List of trade records for trade-specific metrics.
            total_api_cost: Total LLM API cost in USD.
            benchmark_return: Buy-and-hold cumulative return for CAA calculation.
            risk_free_rate: Annualized risk-free rate.
        """
        if len(portfolio_values) < 2:
            return self._empty_metrics()

        values = np.array(portfolio_values, dtype=np.float64)
        initial = values[0]
        final = values[-1]

        # Daily returns
        returns = np.diff(values) / values[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) == 0:
            return self._empty_metrics()

        # ── Return Metrics ───────────────────────────────────────
        cumulative_return = (final / initial) - 1.0
        n_days = len(returns)
        annualized_return = (1 + cumulative_return) ** (TRADING_DAYS_PER_YEAR / max(n_days, 1)) - 1

        # ── Risk Metrics ─────────────────────────────────────────
        volatility = float(np.std(returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))

        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        excess_returns = returns - daily_rf

        # Sharpe Ratio
        sharpe_ratio = 0.0
        if np.std(excess_returns) > 0:
            sharpe_ratio = float(
                np.mean(excess_returns) / np.std(excess_returns, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
            )

        # Sortino Ratio (downside deviation only)
        downside = excess_returns[excess_returns < 0]
        sortino_ratio = 0.0
        if len(downside) > 0 and np.std(downside) > 0:
            sortino_ratio = float(
                np.mean(excess_returns) / np.std(downside, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR)
            )

        # Maximum Drawdown
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        max_drawdown_pct = float(np.min(drawdown))
        max_drawdown = float(np.min(values - peak))

        # Calmar Ratio
        calmar_ratio = 0.0
        if abs(max_drawdown_pct) > 0:
            calmar_ratio = annualized_return / abs(max_drawdown_pct)

        # ── Trading Metrics ──────────────────────────────────────
        trade_list = trades or []
        trade_count = len(trade_list)

        winning = [t for t in trade_list if t.get("realized_pnl", 0) > 0]
        losing = [t for t in trade_list if t.get("realized_pnl", 0) < 0]

        win_rate = len(winning) / trade_count if trade_count > 0 else 0.0

        gross_profit = sum(t.get("realized_pnl", 0) for t in winning)
        gross_loss = abs(sum(t.get("realized_pnl", 0) for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        avg_holding_period = 0.0  # Placeholder — requires position tracking

        # ── AI-Specific Metrics ──────────────────────────────────
        signal_count = max(trade_count, 1)
        cost_per_signal = total_api_cost / signal_count

        # Cost-Adjusted Alpha
        cost_adjusted_alpha = 0.0
        if total_api_cost > 0:
            cost_adjusted_alpha = (cumulative_return - benchmark_return) / total_api_cost

        signal_accuracy = 0.0  # Requires post-hoc direction comparison

        return PerformanceMetrics(
            cumulative_return=round(cumulative_return, 4),
            annualized_return=round(annualized_return, 4),
            sharpe_ratio=round(sharpe_ratio, 2),
            sortino_ratio=round(sortino_ratio, 2),
            max_drawdown=round(max_drawdown, 0),
            max_drawdown_pct=round(max_drawdown_pct, 4),
            calmar_ratio=round(calmar_ratio, 2),
            volatility=round(volatility, 4),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2),
            trade_count=trade_count,
            avg_holding_period_days=round(avg_holding_period, 1),
            cost_per_signal=round(cost_per_signal, 6),
            cost_adjusted_alpha=round(cost_adjusted_alpha, 4),
            signal_accuracy=round(signal_accuracy, 4),
        )

    def _empty_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            cumulative_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
            sortino_ratio=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
            calmar_ratio=0.0, volatility=0.0, win_rate=0.0, profit_factor=0.0,
            trade_count=0, avg_holding_period_days=0.0, cost_per_signal=0.0,
            cost_adjusted_alpha=0.0, signal_accuracy=0.0,
        )
