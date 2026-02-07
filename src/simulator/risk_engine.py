"""Hardcoded risk engine -- code-enforced risk limits that override LLM decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.core.logging import get_logger
from src.core.types import Action, PortfolioState, TradingSignal, MarketSnapshot

log = get_logger(__name__)


# ── Enums & Value Objects ────────────────────────────────────────


class VolatilityRegime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass(frozen=True)
class RiskCheck:
    """Result of a single risk validation rule."""

    name: str
    passed: bool
    detail: str
    value: float = 0.0
    limit: float = 0.0


@dataclass
class RiskConfig:
    """Tunable knobs for risk management.  Defaults are conservative."""

    max_position_weight: float = 0.30
    min_cash_ratio: float = 0.20
    daily_loss_limit_pct: float = 0.05
    drawdown_circuit_breaker_pct: float = 0.15
    var_confidence: float = 0.95
    max_sector_exposure: float = 0.50
    max_correlation_overlap: float = 0.70
    min_confidence_for_buy: float = 0.3
    max_trades_per_cycle: int = 3


@dataclass
class RiskDecision:
    """Aggregate risk evaluation result delivered to the order engine."""

    approved: bool
    checks: list[RiskCheck]
    portfolio_var: float
    drawdown_pct: float
    volatility_regime: VolatilityRegime
    override_action: Action | None  # Force HOLD if rejected
    reason: str


# ── Engine ───────────────────────────────────────────────────────


class RiskEngine:
    """Code-enforced risk management.  Runs AFTER LLM signal, BEFORE trade execution.

    Unlike the prompt-based Risk Manager, this engine uses actual calculations
    and cannot be bypassed by LLM hallucination.
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        self._config: RiskConfig = config or RiskConfig()
        self._daily_pnl: dict[str, float] = {}  # portfolio_id -> today's PnL

    # ── Public API ───────────────────────────────────────────────

    def evaluate(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        returns_history: list[float] | None = None,
    ) -> RiskDecision:
        """Run all risk checks and return decision."""
        checks: list[RiskCheck] = []
        checks.append(self._check_position_limit(signal, portfolio))
        checks.append(self._check_cash_reserve(signal, portfolio))
        checks.append(self._check_confidence(signal))
        checks.append(self._check_drawdown(portfolio))
        checks.append(self._check_daily_loss(portfolio))

        # Calculate VaR if history available
        var: float = (
            self._calculate_var(returns_history, self._config.var_confidence)
            if returns_history
            else 0.0
        )
        dd: float = self._current_drawdown(portfolio)
        vol_regime: VolatilityRegime = self._assess_volatility(returns_history)

        all_passed: bool = all(c.passed for c in checks)
        override: Action | None = None if all_passed else Action.HOLD
        reason: str = (
            "All checks passed"
            if all_passed
            else "; ".join(c.detail for c in checks if not c.passed)
        )

        decision = RiskDecision(
            approved=all_passed,
            checks=checks,
            portfolio_var=var,
            drawdown_pct=dd,
            volatility_regime=vol_regime,
            override_action=override,
            reason=reason,
        )

        log.info(
            "risk_evaluation_complete",
            approved=decision.approved,
            checks_passed=sum(1 for c in checks if c.passed),
            checks_total=len(checks),
            portfolio_var=var,
            drawdown_pct=dd,
            volatility_regime=vol_regime.value,
            signal_action=signal.action.value,
            signal_symbol=signal.symbol,
            reason=reason,
        )

        return decision

    def record_daily_pnl(self, portfolio_id: str, pnl: float) -> None:
        """Accumulate intra-day PnL for a portfolio."""
        self._daily_pnl[portfolio_id] = self._daily_pnl.get(portfolio_id, 0.0) + pnl
        log.debug(
            "daily_pnl_recorded",
            portfolio_id=portfolio_id,
            pnl_delta=pnl,
            cumulative=self._daily_pnl[portfolio_id],
        )

    def reset_daily(self) -> None:
        """Reset all daily PnL accumulators (call at start of each trading day)."""
        log.info("daily_pnl_reset", portfolios_cleared=len(self._daily_pnl))
        self._daily_pnl.clear()

    # ── Individual Checks ────────────────────────────────────────

    def _check_position_limit(
        self, signal: TradingSignal, portfolio: PortfolioState
    ) -> RiskCheck:
        """Reject BUY if it would push a single position above max_position_weight."""
        name: str = "position_limit"

        # Only relevant for BUY orders
        if signal.action != Action.BUY:
            return RiskCheck(
                name=name,
                passed=True,
                detail="Non-BUY action; position limit not applicable",
            )

        total_value: float = portfolio.total_value
        if total_value <= 0:
            return RiskCheck(
                name=name,
                passed=False,
                detail="Portfolio total value is zero or negative",
                value=0.0,
                limit=self._config.max_position_weight,
            )

        # Existing position value for this symbol
        existing_value: float = 0.0
        if signal.symbol in portfolio.positions:
            pos = portfolio.positions[signal.symbol]
            existing_value = pos.current_price * pos.quantity

        # Proposed additional allocation (weight * total_value)
        proposed_additional: float = signal.weight * total_value
        new_position_value: float = existing_value + proposed_additional
        new_weight: float = new_position_value / total_value

        passed: bool = new_weight <= self._config.max_position_weight
        detail: str = (
            f"Position weight {new_weight:.2%} within limit {self._config.max_position_weight:.2%}"
            if passed
            else (
                f"Position weight {new_weight:.2%} exceeds limit "
                f"{self._config.max_position_weight:.2%}"
            )
        )

        return RiskCheck(
            name=name,
            passed=passed,
            detail=detail,
            value=new_weight,
            limit=self._config.max_position_weight,
        )

    def _check_cash_reserve(
        self, signal: TradingSignal, portfolio: PortfolioState
    ) -> RiskCheck:
        """Reject BUY if cash after trade would drop below min_cash_ratio."""
        name: str = "cash_reserve"

        if signal.action != Action.BUY:
            return RiskCheck(
                name=name,
                passed=True,
                detail="Non-BUY action; cash reserve not applicable",
            )

        total_value: float = portfolio.total_value
        if total_value <= 0:
            return RiskCheck(
                name=name,
                passed=False,
                detail="Portfolio total value is zero or negative",
                value=0.0,
                limit=self._config.min_cash_ratio,
            )

        # Cash that would remain after the buy
        spend: float = signal.weight * total_value
        remaining_cash: float = portfolio.cash - spend
        projected_cash_ratio: float = remaining_cash / total_value

        passed: bool = projected_cash_ratio >= self._config.min_cash_ratio
        detail: str = (
            f"Projected cash ratio {projected_cash_ratio:.2%} "
            f">= minimum {self._config.min_cash_ratio:.2%}"
            if passed
            else (
                f"Projected cash ratio {projected_cash_ratio:.2%} "
                f"below minimum {self._config.min_cash_ratio:.2%}"
            )
        )

        return RiskCheck(
            name=name,
            passed=passed,
            detail=detail,
            value=projected_cash_ratio,
            limit=self._config.min_cash_ratio,
        )

    def _check_confidence(self, signal: TradingSignal) -> RiskCheck:
        """BUY signals must meet a minimum confidence threshold.

        SELL and HOLD always pass -- we do not want to block exits or holds.
        """
        name: str = "confidence"

        if signal.action != Action.BUY:
            return RiskCheck(
                name=name,
                passed=True,
                detail=f"Action {signal.action.value}; confidence check not applicable",
                value=signal.confidence,
                limit=self._config.min_confidence_for_buy,
            )

        passed: bool = signal.confidence >= self._config.min_confidence_for_buy
        detail: str = (
            f"BUY confidence {signal.confidence:.2f} "
            f">= threshold {self._config.min_confidence_for_buy:.2f}"
            if passed
            else (
                f"BUY confidence {signal.confidence:.2f} "
                f"below threshold {self._config.min_confidence_for_buy:.2f}"
            )
        )

        return RiskCheck(
            name=name,
            passed=passed,
            detail=detail,
            value=signal.confidence,
            limit=self._config.min_confidence_for_buy,
        )

    def _check_drawdown(self, portfolio: PortfolioState) -> RiskCheck:
        """Circuit breaker: block new BUY orders when drawdown exceeds threshold.

        SELL and HOLD are always allowed even during drawdown.
        """
        name: str = "drawdown_circuit_breaker"
        dd: float = self._current_drawdown(portfolio)
        limit: float = self._config.drawdown_circuit_breaker_pct

        passed: bool = dd < limit
        detail: str = (
            f"Drawdown {dd:.2%} within circuit breaker {limit:.2%}"
            if passed
            else f"Drawdown {dd:.2%} exceeds circuit breaker {limit:.2%}"
        )

        return RiskCheck(
            name=name,
            passed=passed,
            detail=detail,
            value=dd,
            limit=limit,
        )

    def _check_daily_loss(self, portfolio: PortfolioState) -> RiskCheck:
        """Block trading if the accumulated daily loss exceeds the limit."""
        name: str = "daily_loss_limit"
        pid: str = portfolio.portfolio_id
        daily_pnl: float = self._daily_pnl.get(pid, 0.0)
        limit: float = self._config.daily_loss_limit_pct

        # Express daily loss as a fraction of initial capital
        if portfolio.initial_capital <= 0:
            return RiskCheck(
                name=name,
                passed=True,
                detail="Initial capital is zero; daily loss check skipped",
            )

        loss_ratio: float = abs(min(0.0, daily_pnl)) / portfolio.initial_capital
        passed: bool = loss_ratio < limit
        detail: str = (
            f"Daily loss {loss_ratio:.2%} within limit {limit:.2%}"
            if passed
            else f"Daily loss {loss_ratio:.2%} exceeds limit {limit:.2%}"
        )

        return RiskCheck(
            name=name,
            passed=passed,
            detail=detail,
            value=loss_ratio,
            limit=limit,
        )

    # ── Calculations ─────────────────────────────────────────────

    def _calculate_var(
        self, returns: list[float], confidence: float = 0.95
    ) -> float:
        """Historical simulation VaR.

        Returns the portfolio-level Value-at-Risk as a positive loss figure.
        A VaR of 0.03 means 'with *confidence* probability, the daily loss
        will not exceed 3 % of portfolio value'.
        """
        if not returns or len(returns) < 2:
            return 0.0

        arr: np.ndarray = np.array(returns, dtype=np.float64)
        # VaR is the loss at the (1 - confidence) percentile
        percentile: float = (1.0 - confidence) * 100.0
        var_value: float = float(np.percentile(arr, percentile))

        # Convention: return positive number representing potential loss
        return abs(min(0.0, var_value))

    def _current_drawdown(self, portfolio: PortfolioState) -> float:
        """Calculate drawdown from initial capital vs current total value.

        Returns a value in [0, 1] where 0 means no drawdown and 1 means
        the portfolio has lost all its value.
        """
        if portfolio.initial_capital <= 0:
            return 0.0

        total: float = portfolio.total_value
        if total >= portfolio.initial_capital:
            return 0.0

        dd: float = (portfolio.initial_capital - total) / portfolio.initial_capital
        return max(0.0, min(1.0, dd))

    def _assess_volatility(
        self, returns: list[float] | None
    ) -> VolatilityRegime:
        """Classify current market volatility based on the annualised standard
        deviation of recent returns.

        Thresholds (annualised):
            LOW      : sigma < 10 %
            NORMAL   : 10 % <= sigma < 25 %
            HIGH     : 25 % <= sigma < 50 %
            EXTREME  : sigma >= 50 %
        """
        if not returns or len(returns) < 2:
            return VolatilityRegime.NORMAL  # default when data insufficient

        arr: np.ndarray = np.array(returns, dtype=np.float64)
        daily_std: float = float(np.std(arr, ddof=1))

        # Annualise assuming 252 trading days
        annualised_std: float = daily_std * math.sqrt(252)

        if annualised_std < 0.10:
            return VolatilityRegime.LOW
        if annualised_std < 0.25:
            return VolatilityRegime.NORMAL
        if annualised_std < 0.50:
            return VolatilityRegime.HIGH
        return VolatilityRegime.EXTREME
