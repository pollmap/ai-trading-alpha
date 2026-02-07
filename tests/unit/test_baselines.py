"""Tests for baseline trading strategies."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from uuid_extensions import uuid7

from src.core.types import (
    Action,
    AgentArchitecture,
    MacroData,
    Market,
    MarketSnapshot,
    ModelProvider,
    PortfolioState,
    Position,
    SymbolData,
)
from src.simulator.baselines import (
    BuyAndHoldBaseline,
    EqualWeightRebalanceBaseline,
    MeanReversionBaseline,
    MomentumBaseline,
    RandomBaseline,
)


@pytest.fixture
def snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id="snap-1",
        timestamp=datetime.now(timezone.utc),
        market=Market.CRYPTO,
        symbols={
            "BTCUSDT": SymbolData(
                symbol="BTCUSDT", market=Market.CRYPTO,
                open=50_000.0, high=51_000.0, low=49_000.0, close=50_500.0,
                volume=1_000_000.0, currency="USDT",
            ),
            "ETHUSDT": SymbolData(
                symbol="ETHUSDT", market=Market.CRYPTO,
                open=3_000.0, high=3_100.0, low=2_900.0, close=2_950.0,
                volume=500_000.0, currency="USDT",
            ),
        },
        macro=MacroData(),
    )


@pytest.fixture
def empty_portfolio() -> PortfolioState:
    return PortfolioState(
        portfolio_id="baseline-test",
        model=ModelProvider.GPT,
        architecture=AgentArchitecture.SINGLE,
        market=Market.CRYPTO,
        cash=100_000.0,
        positions={},
        initial_capital=100_000.0,
    )


class TestBuyAndHold:
    def test_first_cycle_buys(self, snapshot: MarketSnapshot, empty_portfolio: PortfolioState) -> None:
        strategy = BuyAndHoldBaseline()
        signals = strategy.generate_signals(snapshot, empty_portfolio)
        assert len(signals) == 2
        assert all(s.action == Action.BUY for s in signals)

    def test_subsequent_cycles_hold(self, snapshot: MarketSnapshot, empty_portfolio: PortfolioState) -> None:
        strategy = BuyAndHoldBaseline()
        strategy.generate_signals(snapshot, empty_portfolio)  # first cycle
        signals = strategy.generate_signals(snapshot, empty_portfolio)  # second cycle
        assert all(s.action == Action.HOLD for s in signals)


class TestEqualWeightRebalance:
    def test_initial_buy(self, snapshot: MarketSnapshot, empty_portfolio: PortfolioState) -> None:
        strategy = EqualWeightRebalanceBaseline()
        signals = strategy.generate_signals(snapshot, empty_portfolio)
        assert len(signals) == 2
        buy_signals = [s for s in signals if s.action == Action.BUY]
        assert len(buy_signals) > 0


class TestMomentum:
    def test_momentum_signals(self, snapshot: MarketSnapshot, empty_portfolio: PortfolioState) -> None:
        strategy = MomentumBaseline(top_n=1)
        signals = strategy.generate_signals(snapshot, empty_portfolio)
        assert len(signals) == 2
        buy_signals = [s for s in signals if s.action == Action.BUY]
        assert len(buy_signals) >= 1

    def test_empty_snapshot(self, empty_portfolio: PortfolioState) -> None:
        strategy = MomentumBaseline()
        snapshot = MarketSnapshot(
            snapshot_id="empty",
            timestamp=datetime.now(timezone.utc),
            market=Market.CRYPTO,
            symbols={},
            macro=MacroData(),
        )
        signals = strategy.generate_signals(snapshot, empty_portfolio)
        assert signals == []


class TestMeanReversion:
    def test_mean_reversion_signals(self, snapshot: MarketSnapshot, empty_portfolio: PortfolioState) -> None:
        strategy = MeanReversionBaseline()
        signals = strategy.generate_signals(snapshot, empty_portfolio)
        assert len(signals) == 2
        assert all(s.reasoning for s in signals)


class TestRandom:
    def test_deterministic_with_seed(self, snapshot: MarketSnapshot, empty_portfolio: PortfolioState) -> None:
        s1 = RandomBaseline(seed=42)
        s2 = RandomBaseline(seed=42)
        signals1 = s1.generate_signals(snapshot, empty_portfolio)
        signals2 = s2.generate_signals(snapshot, empty_portfolio)
        assert len(signals1) == len(signals2)
        for a, b in zip(signals1, signals2):
            assert a.action == b.action

    def test_all_have_reasoning(self, snapshot: MarketSnapshot, empty_portfolio: PortfolioState) -> None:
        strategy = RandomBaseline(seed=0)
        signals = strategy.generate_signals(snapshot, empty_portfolio)
        assert all(s.reasoning.strip() for s in signals)
