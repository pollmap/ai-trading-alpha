"""Tests for execution timing optimizer."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from uuid_extensions import uuid7

from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    ModelProvider,
    TradingSignal,
)
from src.rl.execution_timer import (
    ExecutionTimer,
    TimingAction,
    TimingDecision,
    TimingOutcome,
)


@pytest.fixture
def timer() -> ExecutionTimer:
    return ExecutionTimer()


def _make_signal(
    action: Action = Action.BUY,
    confidence: float = 0.7,
) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id="snap-1",
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        market=Market.CRYPTO,
        action=action,
        weight=0.15,
        confidence=confidence,
        reasoning="Timer test signal",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
    )


class TestExecutionTimer:
    def test_default_immediate(self, timer: ExecutionTimer) -> None:
        signal = _make_signal()
        decision = timer.decide(signal)
        assert decision.action == TimingAction.IMMEDIATE
        assert decision.execute_pct == 1.0

    def test_high_confidence_sell_immediate(self, timer: ExecutionTimer) -> None:
        signal = _make_signal(Action.SELL, confidence=0.9)
        decision = timer.decide(signal)
        assert decision.action == TimingAction.IMMEDIATE

    def test_high_vol_low_conf_waits(self, timer: ExecutionTimer) -> None:
        signal = _make_signal(Action.BUY, confidence=0.4)
        decision = timer.decide(signal, current_volatility=0.05)
        assert decision.action == TimingAction.WAIT_1_CYCLE
        assert decision.execute_pct == 0.0

    def test_strong_momentum_buy_immediate(self, timer: ExecutionTimer) -> None:
        signal = _make_signal(Action.BUY)
        decision = timer.decide(signal, recent_momentum=0.05)
        assert decision.action == TimingAction.IMMEDIATE

    def test_strong_momentum_sell_immediate(self, timer: ExecutionTimer) -> None:
        signal = _make_signal(Action.SELL)
        decision = timer.decide(signal, recent_momentum=-0.05)
        assert decision.action == TimingAction.IMMEDIATE

    def test_wide_spread_splits(self, timer: ExecutionTimer) -> None:
        signal = _make_signal(Action.BUY)
        decision = timer.decide(signal, spread_bps=100)
        assert decision.action == TimingAction.SPLIT_EXECUTION
        assert decision.execute_pct == 0.5

    def test_record_outcome(self, timer: ExecutionTimer) -> None:
        signal = _make_signal()
        decision = TimingDecision(
            action=TimingAction.IMMEDIATE,
            confidence=0.8,
            reason="test",
            execute_pct=1.0,
        )
        outcome = TimingOutcome(
            decision=decision,
            signal=signal,
            decision_price=50_000.0,
            execution_price=50_050.0,
            optimal_price=49_980.0,
            slippage_saved=-50.0,
        )
        timer.record_outcome(outcome)
        assert timer.outcome_count == 1
        assert timer.avg_slippage_saved == -50.0

    def test_empty_outcomes(self, timer: ExecutionTimer) -> None:
        assert timer.avg_slippage_saved == 0.0
        assert timer.outcome_count == 0
