"""Tests for agent self-reflection."""

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
from src.agents.reflection import AgentReflector, TradeOutcome


@pytest.fixture
def reflector() -> AgentReflector:
    return AgentReflector(lookback=10)


def _make_signal(action: Action = Action.BUY) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id="snap-1",
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        market=Market.CRYPTO,
        action=action,
        weight=0.15,
        confidence=0.8,
        reasoning="Test signal for reflection",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
    )


def _make_outcome(was_correct: bool, pnl: float = 100.0, action: Action = Action.BUY) -> TradeOutcome:
    return TradeOutcome(
        signal=_make_signal(action),
        entry_price=50_000.0,
        current_price=50_500.0 if was_correct else 49_500.0,
        realized_pnl=pnl if was_correct else -abs(pnl),
        holding_periods=3,
        was_correct=was_correct,
    )


class TestAgentReflector:
    def test_empty_history(self, reflector: AgentReflector) -> None:
        text = reflector.generate_reflection("test_key")
        assert text == ""

    def test_insufficient_history(self, reflector: AgentReflector) -> None:
        reflector.record_outcome("key", _make_outcome(True))
        reflector.record_outcome("key", _make_outcome(False))
        text = reflector.generate_reflection("key")
        assert text == ""  # needs 3 minimum

    def test_generates_reflection(self, reflector: AgentReflector) -> None:
        for i in range(5):
            reflector.record_outcome("key", _make_outcome(i % 2 == 0))
        text = reflector.generate_reflection("key")
        assert "Self-Reflection" in text
        assert "Win Rate" in text

    def test_overconfidence_detection(self, reflector: AgentReflector) -> None:
        # All losses but high confidence signals
        for _ in range(5):
            reflector.record_outcome("key", _make_outcome(False, 100.0))
        text = reflector.generate_reflection("key")
        assert "CALIBRATION" in text or "inaccurate" in text.lower()

    def test_clear_history(self, reflector: AgentReflector) -> None:
        reflector.record_outcome("key", _make_outcome(True))
        reflector.clear_history("key")
        assert reflector.get_outcome_count("key") == 0

    def test_outcome_count(self, reflector: AgentReflector) -> None:
        for _ in range(5):
            reflector.record_outcome("key", _make_outcome(True))
        assert reflector.get_outcome_count("key") == 5

    def test_history_pruning(self, reflector: AgentReflector) -> None:
        # Reflector with lookback=10, prunes at 2*lookback=20
        # After 21 records, pruning kicks in -> trims to 10
        # Then we add 4 more -> 14 total until next prune
        for _ in range(25):
            reflector.record_outcome("key", _make_outcome(True))
        # After 25 records: pruned at 21 to 10, then 4 more added = 14
        assert reflector.get_outcome_count("key") <= 20  # bounded by 2*lookback
