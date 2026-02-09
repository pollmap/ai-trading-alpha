"""Tests for custom strategy system."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.agents.custom_strategy import (
    StrategyTemplate,
    StrategyBuilder,
    StrategyStore,
    StrategyStatus,
    RiskParameters,
    TEMPLATE_VARIABLES,
)
from src.core.types import (
    Market,
    ModelProvider,
    MarketSnapshot,
    PortfolioState,
    AgentArchitecture,
    MacroData,
    SymbolData,
)


def _make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id="snap-001",
        timestamp=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        market=Market.US,
        symbols={
            "AAPL": SymbolData(
                symbol="AAPL",
                market=Market.US,
                open=185.0,
                high=188.0,
                low=184.0,
                close=187.0,
                volume=50_000_000,
                currency="USD",
            ),
        },
        macro=MacroData(us_fed_rate=5.25, vix=18.5),
    )


def _make_portfolio() -> PortfolioState:
    return PortfolioState(
        portfolio_id="port-001",
        model=ModelProvider.CLAUDE,
        architecture=AgentArchitecture.SINGLE,
        market=Market.US,
        cash=80_000.0,
        positions={},
        initial_capital=100_000.0,
    )


def _make_template(**kwargs) -> StrategyTemplate:
    defaults = {
        "strategy_id": "strat-001",
        "name": "Momentum Scanner",
        "description": "Buy on strong momentum signals",
        "owner_id": "user-123",
        "prompt_template": (
            "Analyze the following market data:\n{market_data}\n\n"
            "Current portfolio:\n{portfolio_state}\n\n"
            "Market regime: {regime}\n\n"
            "Make a trading decision."
        ),
        "model": ModelProvider.CLAUDE,
        "markets": [Market.US],
    }
    defaults.update(kwargs)
    return StrategyTemplate(**defaults)


class TestRiskParameters:
    """Tests for RiskParameters validation."""

    def test_valid_defaults(self) -> None:
        params = RiskParameters()
        errors = params.validate()
        assert errors == []

    def test_invalid_max_position_weight(self) -> None:
        params = RiskParameters(max_position_weight=1.5)
        errors = params.validate()
        assert any("max_position_weight" in e for e in errors)

    def test_invalid_stop_loss(self) -> None:
        params = RiskParameters(stop_loss_pct=-0.05)
        errors = params.validate()
        assert any("stop_loss_pct" in e for e in errors)

    def test_invalid_position_size_mode(self) -> None:
        params = RiskParameters(position_size_mode="invalid")
        errors = params.validate()
        assert any("position_size_mode" in e for e in errors)


class TestStrategyBuilder:
    """Tests for StrategyBuilder."""

    def test_validate_valid_template(self) -> None:
        builder = StrategyBuilder()
        template = _make_template()
        errors = builder.validate_template(template)
        assert errors == []

    def test_validate_empty_name(self) -> None:
        builder = StrategyBuilder()
        template = _make_template(name="")
        errors = builder.validate_template(template)
        assert any("name" in e.lower() for e in errors)

    def test_validate_empty_prompt(self) -> None:
        builder = StrategyBuilder()
        template = _make_template(prompt_template="")
        errors = builder.validate_template(template)
        assert any("prompt" in e.lower() for e in errors)

    def test_validate_unknown_variables(self) -> None:
        builder = StrategyBuilder()
        template = _make_template(
            prompt_template="Use {market_data} and {unknown_var} to decide."
        )
        errors = builder.validate_template(template)
        assert any("unknown" in e.lower() for e in errors)

    def test_validate_no_markets(self) -> None:
        builder = StrategyBuilder()
        template = _make_template(markets=[])
        errors = builder.validate_template(template)
        assert any("market" in e.lower() for e in errors)

    def test_compile_prompt(self) -> None:
        builder = StrategyBuilder()
        template = _make_template()
        snapshot = _make_snapshot()
        portfolio = _make_portfolio()

        prompt = builder.compile_prompt(template, snapshot, portfolio, regime="bull")

        assert "Momentum Scanner" in prompt
        assert "AAPL" in prompt
        assert "bull" in prompt
        assert "BUY" in prompt and "SELL" in prompt  # response instructions

    def test_compile_with_custom_indicators(self) -> None:
        builder = StrategyBuilder()
        template = _make_template(
            prompt_template="Indicators: {custom_indicators}\nData: {market_data}"
        )
        prompt = builder.compile_prompt(
            template,
            _make_snapshot(),
            _make_portfolio(),
            custom_indicators={"rsi": 72.5, "macd": 1.23},
        )
        assert "rsi=72.5000" in prompt
        assert "macd=1.2300" in prompt

    def test_all_template_variables(self) -> None:
        """Ensure all documented variables are available."""
        expected = {"market_data", "portfolio_state", "regime", "news", "macro", "custom_indicators"}
        assert set(TEMPLATE_VARIABLES) == expected


class TestStrategyStore:
    """Tests for StrategyStore CRUD."""

    def test_create_and_get(self) -> None:
        store = StrategyStore()
        template = _make_template()
        created = store.create(template)
        assert created.strategy_id == "strat-001"

        retrieved = store.get("strat-001")
        assert retrieved is not None
        assert retrieved.name == "Momentum Scanner"

    def test_list_by_owner(self) -> None:
        store = StrategyStore()
        store.create(_make_template(strategy_id="s1", owner_id="user-A"))
        store.create(_make_template(strategy_id="s2", owner_id="user-A"))
        store.create(_make_template(strategy_id="s3", owner_id="user-B"))

        user_a = store.list_by_owner("user-A")
        assert len(user_a) == 2

        user_b = store.list_by_owner("user-B")
        assert len(user_b) == 1

    def test_update_increments_version(self) -> None:
        store = StrategyStore()
        template = _make_template()
        store.create(template)

        template.description = "Updated description"
        updated = store.update(template)
        assert updated.version == 2
        assert updated.description == "Updated description"

    def test_delete(self) -> None:
        store = StrategyStore()
        store.create(_make_template())
        assert store.delete("strat-001") is True
        assert store.get("strat-001") is None
        assert store.delete("strat-001") is False

    def test_activate(self) -> None:
        store = StrategyStore()
        store.create(_make_template())
        assert store.activate("strat-001") is True
        assert store.get("strat-001").status == StrategyStatus.ACTIVE

    def test_archive(self) -> None:
        store = StrategyStore()
        store.create(_make_template())
        assert store.archive("strat-001") is True
        assert store.get("strat-001").status == StrategyStatus.ARCHIVED
