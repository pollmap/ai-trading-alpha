"""Tests for portfolio manager and position tracker."""

from __future__ import annotations

import pytest

from src.core.exceptions import InsufficientFundsError
from src.core.types import (
    AgentArchitecture,
    Market,
    ModelProvider,
)
from src.simulator.portfolio import PortfolioManager
from src.simulator.position_tracker import PositionTracker


class TestPositionTracker:
    def test_open_position(self) -> None:
        tracker = PositionTracker()
        pos = tracker.open_position("BTCUSDT", 1.0, 50_000.0)
        assert pos.symbol == "BTCUSDT"
        assert pos.quantity == 1.0
        assert pos.avg_entry_price == 50_000.0
        assert len(tracker.get_history()) == 1

    def test_add_to_position(self) -> None:
        tracker = PositionTracker()
        pos = tracker.open_position("BTCUSDT", 1.0, 50_000.0)
        pos = tracker.add_to_position(pos, 1.0, 60_000.0)
        assert pos.quantity == 2.0
        assert pos.avg_entry_price == 55_000.0  # weighted average
        assert len(tracker.get_history()) == 2

    def test_reduce_position(self) -> None:
        tracker = PositionTracker()
        pos = tracker.open_position("BTCUSDT", 2.0, 50_000.0)
        pos, realized = tracker.reduce_position(pos, 1.0, 60_000.0)
        assert pos.quantity == 1.0
        assert realized == 10_000.0  # (60k - 50k) * 1
        assert len(tracker.get_history()) == 2

    def test_close_position(self) -> None:
        tracker = PositionTracker()
        pos = tracker.open_position("BTCUSDT", 1.0, 50_000.0)
        pos, realized = tracker.reduce_position(pos, 1.0, 55_000.0)
        assert pos.quantity == 0.0
        assert realized == 5_000.0

    def test_cannot_sell_more_than_held(self) -> None:
        tracker = PositionTracker()
        pos = tracker.open_position("BTCUSDT", 1.0, 50_000.0)
        with pytest.raises(ValueError, match="only 1.0 held"):
            tracker.reduce_position(pos, 2.0, 50_000.0)

    def test_open_invalid_quantity(self) -> None:
        tracker = PositionTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.open_position("BTCUSDT", -1.0, 50_000.0)

    def test_open_invalid_price(self) -> None:
        tracker = PositionTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.open_position("BTCUSDT", 1.0, 0.0)


class TestPortfolioManager:
    def test_init_portfolios_creates_nine(self) -> None:
        mgr = PortfolioManager()
        ids = mgr.init_portfolios(Market.CRYPTO, 100_000)
        assert len(ids) == 9  # 4 models x 2 architectures + 1 buy&hold

    def test_get_state(self) -> None:
        mgr = PortfolioManager()
        mgr.init_portfolios(Market.US, 100_000)
        state = mgr.get_state(ModelProvider.CLAUDE, AgentArchitecture.MULTI, Market.US)
        assert state.model == ModelProvider.CLAUDE
        assert state.architecture == AgentArchitecture.MULTI
        assert state.cash == 100_000

    def test_get_buy_hold_state(self) -> None:
        mgr = PortfolioManager()
        mgr.init_portfolios(Market.CRYPTO, 100_000)
        bh = mgr.get_buy_hold_state(Market.CRYPTO)
        assert bh.cash == 100_000

    def test_get_state_not_initialized(self) -> None:
        mgr = PortfolioManager()
        with pytest.raises(KeyError, match="init_portfolios"):
            mgr.get_state(ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE, Market.KRX)

    def test_update_position_buy(self) -> None:
        mgr = PortfolioManager()
        mgr.init_portfolios(Market.CRYPTO, 100_000)
        state = mgr.get_state(ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE, Market.CRYPTO)
        updated = mgr.update_position(state.portfolio_id, "BTCUSDT", 0.5, 50_000.0)
        assert "BTCUSDT" in updated.positions
        assert updated.cash < 100_000

    def test_update_position_sell(self) -> None:
        mgr = PortfolioManager()
        mgr.init_portfolios(Market.CRYPTO, 100_000)
        state = mgr.get_state(ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE, Market.CRYPTO)
        # Buy first
        updated = mgr.update_position(state.portfolio_id, "BTCUSDT", 1.0, 50_000.0)
        # Then sell
        updated = mgr.update_position(state.portfolio_id, "BTCUSDT", -0.5, 55_000.0)
        assert updated.positions["BTCUSDT"].quantity == 0.5

    def test_insufficient_funds(self) -> None:
        mgr = PortfolioManager()
        mgr.init_portfolios(Market.CRYPTO, 1_000)
        state = mgr.get_state(ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE, Market.CRYPTO)
        with pytest.raises(InsufficientFundsError):
            mgr.update_position(state.portfolio_id, "BTCUSDT", 1.0, 50_000.0)

    def test_snapshot_all(self) -> None:
        mgr = PortfolioManager()
        mgr.init_portfolios(Market.CRYPTO, 100_000)
        mgr.init_portfolios(Market.US, 100_000)
        all_portfolios = mgr.snapshot_all()
        assert len(all_portfolios) == 18  # 9 per market x 2 markets
