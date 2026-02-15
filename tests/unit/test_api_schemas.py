"""Tests for API Pydantic schemas."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.api.models.schemas import (
    ErrorResponse,
    HealthResponse,
    PortfolioOut,
    PositionOut,
    SimulationCreate,
    SimulationOut,
    SimulationStatus,
    StrategyCreate,
    StrategyOut,
    StrategyUpdate,
    UsageOut,
    UserResponse,
)


class TestUserResponse:
    def test_basic(self) -> None:
        user = UserResponse(
            tenant_id="t1",
            name="Alice",
            email="a@b.com",
            plan="free",
        )
        assert user.tenant_id == "t1"
        assert user.avatar_url == ""
        assert user.provider == ""

    def test_full(self) -> None:
        user = UserResponse(
            tenant_id="t2",
            name="Bob",
            email="b@c.com",
            avatar_url="https://img.example.com/bob.jpg",
            plan="pro",
            provider="github",
        )
        assert user.provider == "github"
        assert user.avatar_url.startswith("https://")


class TestSimulationCreate:
    def test_defaults(self) -> None:
        sim = SimulationCreate(name="Test Sim")
        assert sim.markets == ["US"]
        assert sim.models == ["claude"]
        assert sim.architectures == ["single"]
        assert sim.cycles == 10

    def test_custom(self) -> None:
        sim = SimulationCreate(
            name="Multi Market",
            markets=["US", "CRYPTO"],
            models=["claude", "gpt"],
            architectures=["single", "multi"],
            cycles=50,
        )
        assert len(sim.markets) == 2
        assert sim.cycles == 50

    def test_name_required(self) -> None:
        with pytest.raises(Exception):
            SimulationCreate(name="")  # min_length=1

    def test_cycles_bounds(self) -> None:
        with pytest.raises(Exception):
            SimulationCreate(name="X", cycles=0)  # ge=1
        with pytest.raises(Exception):
            SimulationCreate(name="X", cycles=1001)  # le=1000


class TestStrategyCreate:
    def test_basic(self) -> None:
        s = StrategyCreate(
            name="My Strategy",
            prompt_template="Analyze the market data and provide a signal.",
        )
        assert s.model == "claude"
        assert s.markets == ["US"]
        assert s.tags == []

    def test_prompt_too_short(self) -> None:
        with pytest.raises(Exception):
            StrategyCreate(name="S", prompt_template="short")


class TestStrategyUpdate:
    def test_partial(self) -> None:
        u = StrategyUpdate(name="New Name")
        assert u.name == "New Name"
        assert u.description is None
        assert u.prompt_template is None

    def test_empty(self) -> None:
        u = StrategyUpdate()
        assert u.name is None


class TestSimulationOut:
    def test_status_enum(self) -> None:
        now = datetime.now(timezone.utc)
        sim = SimulationOut(
            simulation_id="s1",
            tenant_id="t1",
            name="Test",
            config_json={"markets": ["US"]},
            status=SimulationStatus.RUNNING,
            created_at=now,
        )
        assert sim.status == SimulationStatus.RUNNING
        assert sim.status.value == "running"


class TestStrategyOut:
    def test_roundtrip(self) -> None:
        now = datetime.now(timezone.utc)
        s = StrategyOut(
            strategy_id="st1",
            tenant_id="t1",
            name="Momentum",
            prompt_template="Identify momentum signals",
            model="gpt",
            markets=["US", "CRYPTO"],
            risk_params={"max_weight": 0.3},
            created_at=now,
            updated_at=now,
        )
        assert s.version == 1
        assert s.status == "draft"


class TestPortfolioOut:
    def test_with_positions(self) -> None:
        p = PortfolioOut(
            portfolio_id="p1",
            model="claude",
            architecture="single",
            market="US",
            cash=50000.0,
            total_value=100000.0,
            positions=[
                PositionOut(
                    symbol="AAPL",
                    quantity=100,
                    avg_entry_price=150.0,
                    current_price=180.0,
                    unrealized_pnl=3000.0,
                ),
            ],
        )
        assert len(p.positions) == 1
        assert p.positions[0].symbol == "AAPL"


class TestHealthResponse:
    def test_defaults(self) -> None:
        h = HealthResponse()
        assert h.status == "ok"
        assert h.version == "0.1.0"


class TestUsageOut:
    def test_basic(self) -> None:
        u = UsageOut(
            tenant_id="t1",
            plan="free",
            limits={"max_simulations": 3},
            current={"simulations": 1},
        )
        assert u.limits["max_simulations"] == 3


class TestErrorResponse:
    def test_basic(self) -> None:
        e = ErrorResponse(detail="Not found")
        assert e.detail == "Not found"
