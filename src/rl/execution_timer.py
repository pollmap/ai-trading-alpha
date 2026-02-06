"""Execution timing optimizer — decide whether to execute immediately or wait.

Uses a simple heuristic + learned thresholds:
- If volatility is high and signal is not urgent, wait for pullback
- If momentum is strong and aligned with signal, execute immediately
- Track historical timing decisions and outcomes to improve
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from src.core.logging import get_logger
from src.core.types import Action, TradingSignal

log = get_logger(__name__)


class TimingAction(str, Enum):
    IMMEDIATE = "immediate"
    WAIT_1_CYCLE = "wait_1_cycle"
    WAIT_2_CYCLES = "wait_2_cycles"
    SPLIT_EXECUTION = "split_execution"  # Execute 50% now, 50% next cycle


@dataclass(frozen=True)
class TimingDecision:
    """Output of the execution timer."""
    action: TimingAction
    confidence: float
    reason: str
    execute_pct: float  # Percentage to execute now (1.0 = all, 0.5 = half)


@dataclass
class TimingOutcome:
    """Record of a timing decision's outcome for learning."""
    decision: TimingDecision
    signal: TradingSignal
    decision_price: float
    execution_price: float
    optimal_price: float  # Best price in the wait window
    slippage_saved: float  # Positive = timing helped


class ExecutionTimer:
    """Decide optimal execution timing for trading signals.

    Combines momentum, volatility, and order flow signals to determine
    whether to execute a trade immediately or wait for a better price.
    """

    def __init__(
        self,
        volatility_threshold: float = 0.03,
        momentum_threshold: float = 0.02,
        min_confidence_for_wait: float = 0.5,
    ) -> None:
        self._vol_threshold: float = volatility_threshold
        self._mom_threshold: float = momentum_threshold
        self._min_conf_wait: float = min_confidence_for_wait
        self._outcomes: list[TimingOutcome] = []

    def decide(
        self,
        signal: TradingSignal,
        current_volatility: float = 0.0,
        recent_momentum: float = 0.0,
        spread_bps: float = 0.0,
    ) -> TimingDecision:
        """Determine optimal execution timing.

        Args:
            signal: Trading signal to time.
            current_volatility: Recent intra-day volatility (e.g., ATR/price).
            recent_momentum: Recent price momentum (positive = uptrend).
            spread_bps: Current bid-ask spread in basis points.

        Returns:
            TimingDecision with recommended action.
        """
        # SELL signals with high urgency: always execute immediately
        if signal.action == Action.SELL and signal.confidence > 0.8:
            return TimingDecision(
                action=TimingAction.IMMEDIATE,
                confidence=0.9,
                reason="High-confidence SELL: immediate execution to limit downside",
                execute_pct=1.0,
            )

        # High volatility + BUY: consider waiting for pullback
        if (
            signal.action == Action.BUY
            and current_volatility > self._vol_threshold
            and signal.confidence < self._min_conf_wait
        ):
            return TimingDecision(
                action=TimingAction.WAIT_1_CYCLE,
                confidence=0.6,
                reason=f"High volatility ({current_volatility:.3f}) with moderate confidence; wait for pullback",
                execute_pct=0.0,
            )

        # Strong momentum aligned with signal: execute immediately
        if signal.action == Action.BUY and recent_momentum > self._mom_threshold:
            return TimingDecision(
                action=TimingAction.IMMEDIATE,
                confidence=0.8,
                reason=f"Strong upward momentum ({recent_momentum:+.3f}) aligned with BUY",
                execute_pct=1.0,
            )

        if signal.action == Action.SELL and recent_momentum < -self._mom_threshold:
            return TimingDecision(
                action=TimingAction.IMMEDIATE,
                confidence=0.8,
                reason=f"Strong downward momentum ({recent_momentum:+.3f}) aligned with SELL",
                execute_pct=1.0,
            )

        # Large spread: split execution
        if spread_bps > 50:
            return TimingDecision(
                action=TimingAction.SPLIT_EXECUTION,
                confidence=0.7,
                reason=f"Wide spread ({spread_bps:.0f}bps); split execution to reduce market impact",
                execute_pct=0.5,
            )

        # Default: immediate execution
        return TimingDecision(
            action=TimingAction.IMMEDIATE,
            confidence=0.7,
            reason="Standard conditions; immediate execution",
            execute_pct=1.0,
        )

    def record_outcome(self, outcome: TimingOutcome) -> None:
        """Record a timing outcome for future learning."""
        self._outcomes.append(outcome)
        # Keep bounded
        if len(self._outcomes) > 1000:
            self._outcomes = self._outcomes[-500:]
        log.debug(
            "timing_outcome_recorded",
            action=outcome.decision.action.value,
            slippage_saved=outcome.slippage_saved,
        )

    @property
    def avg_slippage_saved(self) -> float:
        """Average slippage saved by timing decisions."""
        if not self._outcomes:
            return 0.0
        return sum(o.slippage_saved for o in self._outcomes) / len(self._outcomes)

    @property
    def outcome_count(self) -> int:
        return len(self._outcomes)
