"""Tests for RL position sizer."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from uuid_extensions import uuid7

from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    ModelProvider,
    PortfolioState,
    TradingSignal,
)
from src.rl.position_sizer import RLPositionSizer, RLState, SizingDecision, SCALE_ACTIONS


@pytest.fixture
def sizer() -> RLPositionSizer:
    return RLPositionSizer(epsilon=0.0)  # greedy for deterministic tests


@pytest.fixture
def portfolio() -> PortfolioState:
    return PortfolioState(
        portfolio_id="rl-test",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
        market=Market.CRYPTO,
        cash=100_000.0,
        positions={},
        initial_capital=100_000.0,
    )


def _make_signal(weight: float = 0.15, confidence: float = 0.7) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id="snap-1",
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        market=Market.CRYPTO,
        action=Action.BUY,
        weight=weight,
        confidence=confidence,
        reasoning="RL test signal",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
    )


class TestRLPositionSizer:
    def test_basic_decision(self, sizer: RLPositionSizer, portfolio: PortfolioState) -> None:
        signal = _make_signal()
        decision = sizer.decide(signal, portfolio)
        assert isinstance(decision, SizingDecision)
        assert decision.scaled_weight <= 1.0
        assert decision.scale_factor in SCALE_ACTIONS

    def test_q_update(self, sizer: RLPositionSizer) -> None:
        sizer.update("test_state", 0, 1.0, "next_state")
        q_values = sizer._get_q_values("test_state")
        assert q_values[0] > 0  # positive reward should increase Q

    def test_exploration(self, portfolio: PortfolioState) -> None:
        sizer = RLPositionSizer(epsilon=1.0)  # always explore
        signal = _make_signal()
        decision = sizer.decide(signal, portfolio)
        assert decision.exploration

    def test_greedy(self, portfolio: PortfolioState) -> None:
        sizer = RLPositionSizer(epsilon=0.0)
        signal = _make_signal()
        decision = sizer.decide(signal, portfolio)
        assert not decision.exploration

    def test_save_load(self, sizer: RLPositionSizer, portfolio: PortfolioState, tmp_path: Path) -> None:
        signal = _make_signal()
        sizer.decide(signal, portfolio)
        sizer.update("s1", 0, 1.0, "s2")

        save_path = tmp_path / "q_table.json"
        sizer.save(save_path)
        assert save_path.exists()

        new_sizer = RLPositionSizer()
        new_sizer.load(save_path)
        assert new_sizer.q_table_size > 0

    def test_weight_capped(self, sizer: RLPositionSizer, portfolio: PortfolioState) -> None:
        signal = _make_signal(weight=0.95)
        decision = sizer.decide(signal, portfolio)
        assert decision.scaled_weight <= 1.0

    def test_state_key(self) -> None:
        state = RLState(
            volatility_regime="normal",
            confidence_bin="high",
            drawdown_bin="none",
            win_rate_bin="above_avg",
        )
        assert state.key == "normal|high|none|above_avg"
