"""Tests for hardcoded risk engine."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from uuid_extensions import uuid7

from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    ModelProvider,
    PortfolioState,
    Position,
    TradingSignal,
)
from src.simulator.risk_engine import (
    RiskCheck,
    RiskConfig,
    RiskDecision,
    RiskEngine,
    VolatilityRegime,
)


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine()


@pytest.fixture
def portfolio() -> PortfolioState:
    return PortfolioState(
        portfolio_id="test-risk",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
        market=Market.CRYPTO,
        cash=80_000.0,
        positions={
            "BTCUSDT": Position(
                symbol="BTCUSDT",
                quantity=0.4,
                avg_entry_price=50_000.0,
                current_price=50_000.0,
            )
        },
        initial_capital=100_000.0,
    )


def _make_signal(action: Action, weight: float = 0.15, confidence: float = 0.8) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id="snap-1",
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        market=Market.CRYPTO,
        action=action,
        weight=weight,
        confidence=confidence,
        reasoning="Test risk signal",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
    )


class TestRiskEngine:
    def test_hold_always_passes(self, engine: RiskEngine, portfolio: PortfolioState) -> None:
        signal = _make_signal(Action.HOLD)
        decision = engine.evaluate(signal, portfolio)
        assert decision.approved

    def test_sell_always_passes(self, engine: RiskEngine, portfolio: PortfolioState) -> None:
        signal = _make_signal(Action.SELL)
        decision = engine.evaluate(signal, portfolio)
        assert decision.approved

    def test_buy_passes_normal(self, engine: RiskEngine, portfolio: PortfolioState) -> None:
        signal = _make_signal(Action.BUY, weight=0.10, confidence=0.8)
        decision = engine.evaluate(signal, portfolio)
        assert decision.approved

    def test_buy_rejected_low_confidence(self, portfolio: PortfolioState) -> None:
        engine = RiskEngine(RiskConfig(min_confidence_for_buy=0.5))
        signal = _make_signal(Action.BUY, weight=0.10, confidence=0.2)
        decision = engine.evaluate(signal, portfolio)
        assert not decision.approved
        assert decision.override_action == Action.HOLD

    def test_buy_rejected_exceeds_position_limit(self, portfolio: PortfolioState) -> None:
        engine = RiskEngine(RiskConfig(max_position_weight=0.20))
        signal = _make_signal(Action.BUY, weight=0.25)
        decision = engine.evaluate(signal, portfolio)
        # Existing position is 20%, adding 25% would exceed 20% limit
        assert not decision.approved

    def test_drawdown_circuit_breaker(self) -> None:
        engine = RiskEngine(RiskConfig(drawdown_circuit_breaker_pct=0.10))
        portfolio = PortfolioState(
            portfolio_id="test-dd",
            model=ModelProvider.DEEPSEEK,
            architecture=AgentArchitecture.SINGLE,
            market=Market.CRYPTO,
            cash=85_000.0,
            positions={},
            initial_capital=100_000.0,
        )
        signal = _make_signal(Action.BUY)
        decision = engine.evaluate(signal, portfolio)
        assert not decision.approved  # 15% drawdown > 10% limit

    def test_daily_loss_tracking(self, engine: RiskEngine, portfolio: PortfolioState) -> None:
        engine.record_daily_pnl("test-risk", -6000.0)
        signal = _make_signal(Action.BUY)
        decision = engine.evaluate(signal, portfolio)
        assert not decision.approved  # 6% daily loss > 5% limit

    def test_daily_reset(self, engine: RiskEngine) -> None:
        engine.record_daily_pnl("test", -100.0)
        engine.reset_daily()
        # After reset, daily PnL should be cleared

    def test_volatility_assessment(self, engine: RiskEngine) -> None:
        low_vol = [0.001] * 30
        assert engine._assess_volatility(low_vol) == VolatilityRegime.LOW
        high_vol = [0.05 * (-1)**i for i in range(30)]
        assert engine._assess_volatility(high_vol) in (VolatilityRegime.HIGH, VolatilityRegime.EXTREME)

    def test_var_calculation(self, engine: RiskEngine) -> None:
        returns = [-0.01, -0.02, 0.01, -0.03, 0.02, -0.01, 0.005, -0.015, 0.01, -0.005]
        var = engine._calculate_var(returns)
        assert var >= 0.0  # VaR is always positive

    def test_decision_has_all_checks(self, engine: RiskEngine, portfolio: PortfolioState) -> None:
        signal = _make_signal(Action.BUY)
        decision = engine.evaluate(signal, portfolio)
        assert len(decision.checks) == 5
        check_names = {c.name for c in decision.checks}
        assert "position_limit" in check_names
        assert "cash_reserve" in check_names
        assert "confidence" in check_names
        assert "drawdown_circuit_breaker" in check_names
        assert "daily_loss_limit" in check_names
