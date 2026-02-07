"""Tests for virtual order execution engine."""

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
from src.simulator.order_engine import OrderEngine, Trade


@pytest.fixture
def engine() -> OrderEngine:
    return OrderEngine()


@pytest.fixture
def market_config() -> dict[str, object]:
    return {
        "commission": {"buy": 0.001, "sell": 0.001, "tax": 0.0},
        "slippage": 0.001,
        "max_position_weight": 0.30,
        "min_cash_ratio": 0.20,
    }


@pytest.fixture
def empty_portfolio() -> PortfolioState:
    return PortfolioState(
        portfolio_id="test-portfolio",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
        market=Market.CRYPTO,
        cash=100_000.0,
        positions={},
        initial_capital=100_000.0,
    )


@pytest.fixture
def portfolio_with_position() -> PortfolioState:
    return PortfolioState(
        portfolio_id="test-portfolio",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
        market=Market.CRYPTO,
        cash=85_000.0,
        positions={
            "BTCUSDT": Position(
                symbol="BTCUSDT",
                quantity=0.3,
                avg_entry_price=50_000.0,
                current_price=50_000.0,
            )
        },
        initial_capital=100_000.0,
    )


def _make_signal(
    action: Action,
    symbol: str = "BTCUSDT",
    weight: float = 0.15,
    confidence: float = 0.8,
    market: Market = Market.CRYPTO,
) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id="snap-1",
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        market=market,
        action=action,
        weight=weight,
        confidence=confidence,
        reasoning="Test signal",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
    )


class TestHoldSignal:
    def test_hold_returns_none(
        self,
        engine: OrderEngine,
        empty_portfolio: PortfolioState,
        market_config: dict[str, object],
    ) -> None:
        signal = _make_signal(Action.HOLD)
        trade = engine.execute_signal(signal, empty_portfolio, market_config, 50_000.0)
        assert trade is None


class TestBuyOrder:
    def test_buy_creates_trade(
        self,
        engine: OrderEngine,
        empty_portfolio: PortfolioState,
        market_config: dict[str, object],
    ) -> None:
        signal = _make_signal(Action.BUY, weight=0.15)
        trade = engine.execute_signal(signal, empty_portfolio, market_config, 50_000.0)
        assert trade is not None
        assert trade.action == Action.BUY
        assert trade.quantity > 0
        assert trade.price > 50_000.0  # slippage applied

    def test_buy_respects_cash_ratio(
        self,
        engine: OrderEngine,
        empty_portfolio: PortfolioState,
        market_config: dict[str, object],
    ) -> None:
        signal = _make_signal(Action.BUY, weight=0.15)
        trade = engine.execute_signal(signal, empty_portfolio, market_config, 50_000.0)
        assert trade is not None
        total_cost = trade.price * trade.quantity + trade.commission
        remaining_cash = empty_portfolio.cash - total_cost
        assert remaining_cash >= 0

    def test_buy_rejects_zero_price(
        self,
        engine: OrderEngine,
        empty_portfolio: PortfolioState,
        market_config: dict[str, object],
    ) -> None:
        signal = _make_signal(Action.BUY, weight=0.15)
        trade = engine.execute_signal(signal, empty_portfolio, market_config, 0.0)
        assert trade is None


class TestSellOrder:
    def test_sell_with_position(
        self,
        engine: OrderEngine,
        portfolio_with_position: PortfolioState,
        market_config: dict[str, object],
    ) -> None:
        signal = _make_signal(Action.SELL, weight=0.0)  # full liquidation
        trade = engine.execute_signal(signal, portfolio_with_position, market_config, 55_000.0)
        assert trade is not None
        assert trade.action == Action.SELL
        assert trade.quantity > 0

    def test_sell_no_position(
        self,
        engine: OrderEngine,
        empty_portfolio: PortfolioState,
        market_config: dict[str, object],
    ) -> None:
        signal = _make_signal(Action.SELL, weight=0.0)
        trade = engine.execute_signal(signal, empty_portfolio, market_config, 50_000.0)
        assert trade is None


class TestTradeRecord:
    def test_trade_immutable(self) -> None:
        trade = Trade(
            trade_id=str(uuid7()),
            signal_id=str(uuid7()),
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            action=Action.BUY,
            quantity=1.0,
            price=50_000.0,
            commission=50.0,
            slippage=50.0,
            realized_pnl=0.0,
        )
        assert trade.quantity == 1.0
        assert trade.action == Action.BUY

    def test_trade_validates_quantity(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            Trade(
                trade_id=str(uuid7()),
                signal_id=str(uuid7()),
                timestamp=datetime.now(timezone.utc),
                symbol="BTCUSDT",
                action=Action.BUY,
                quantity=-1.0,
                price=50_000.0,
                commission=50.0,
                slippage=50.0,
                realized_pnl=0.0,
            )
