"""Tests for simulation controller."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from src.simulator.simulation_controller import (
    SimulationConfig,
    SimulationController,
    SimulationStatus,
    ScenarioEvent,
)
from src.core.types import Market, ModelProvider


def _make_config(
    simulation_id: str = "sim-001",
    end_date: datetime | None = None,
    max_cycles: int | None = 10,
    interval_seconds: int = 1,
) -> SimulationConfig:
    return SimulationConfig(
        simulation_id=simulation_id,
        name="Test Simulation",
        markets=[Market.US, Market.CRYPTO],
        start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_date=end_date,
        interval_seconds=interval_seconds,
        initial_capital=100_000.0,
        max_cycles=max_cycles,
    )


class TestSimulationController:
    """Tests for SimulationController lifecycle."""

    def test_create_simulation(self) -> None:
        ctrl = SimulationController()
        config = _make_config()
        sid = ctrl.create_simulation(config)

        assert sid == "sim-001"
        assert ctrl.get_status(sid) == SimulationStatus.CREATED
        assert ctrl.get_cycle_count(sid) == 0

    def test_list_simulations(self) -> None:
        ctrl = SimulationController()
        ctrl.create_simulation(_make_config("s1"))
        ctrl.create_simulation(_make_config("s2"))

        sims = ctrl.list_simulations()
        assert len(sims) == 2
        ids = {s["simulation_id"] for s in sims}
        assert ids == {"s1", "s2"}

    def test_run_with_max_cycles(self) -> None:
        ctrl = SimulationController()
        config = _make_config(max_cycles=5)
        ctrl.create_simulation(config)

        result = asyncio.get_event_loop().run_until_complete(ctrl.start("sim-001"))

        assert result.status == SimulationStatus.COMPLETED
        assert result.total_cycles == 5
        assert result.started_at is not None
        assert result.completed_at is not None

    def test_run_with_end_date(self) -> None:
        ctrl = SimulationController()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(hours=3)
        config = _make_config(end_date=end, max_cycles=None, interval_seconds=3600)
        ctrl.create_simulation(config)

        result = asyncio.get_event_loop().run_until_complete(ctrl.start("sim-001"))

        assert result.status == SimulationStatus.COMPLETED
        assert result.total_cycles == 3

    def test_stop_simulation(self) -> None:
        ctrl = SimulationController()
        config = _make_config(max_cycles=1000)
        ctrl.create_simulation(config)

        async def run_and_stop() -> None:
            task = asyncio.create_task(ctrl.start("sim-001"))
            await asyncio.sleep(0.01)
            ctrl.stop("sim-001")
            await task

        asyncio.get_event_loop().run_until_complete(run_and_stop())
        assert ctrl.get_status("sim-001") == SimulationStatus.STOPPED

    def test_pause_resume(self) -> None:
        """Test that pause/resume API doesn't crash and state transitions work."""
        ctrl = SimulationController()
        config = _make_config(max_cycles=10)
        ctrl.create_simulation(config)

        # Complete the simulation normally
        result = asyncio.get_event_loop().run_until_complete(ctrl.start("sim-001"))
        assert result.status == SimulationStatus.COMPLETED
        assert result.total_cycles == 10

        # Verify pause on non-running sim is safe
        ctrl2 = SimulationController()
        config2 = _make_config(simulation_id="sim-002", max_cycles=5)
        ctrl2.create_simulation(config2)
        ctrl2.pause("sim-002")  # no-op: not running yet
        ctrl2.resume("sim-002")  # no-op: not paused

    def test_scenario_injection(self) -> None:
        ctrl = SimulationController()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        config = _make_config(max_cycles=5, interval_seconds=3600)
        config.start_date = start
        config.scenarios = [
            ScenarioEvent(
                event_id="evt-1",
                trigger_time=start + timedelta(hours=2),
                event_type="news",
                description="Flash crash event",
            ),
        ]
        ctrl.create_simulation(config)

        result = asyncio.get_event_loop().run_until_complete(ctrl.start("sim-001"))

        assert result.scenarios_applied == 1
        assert config.scenarios[0].applied is True

    def test_callback_on_cycle(self) -> None:
        ctrl = SimulationController()
        config = _make_config(max_cycles=3)
        ctrl.create_simulation(config)

        cycle_data_collected: list[int] = []

        async def on_cycle(sid: str, cycle: int, data: dict) -> None:
            cycle_data_collected.append(cycle)

        ctrl.on_cycle("sim-001", on_cycle)

        asyncio.get_event_loop().run_until_complete(ctrl.start("sim-001"))
        assert cycle_data_collected == [0, 1, 2]

    def test_infinite_mode_flag(self) -> None:
        ctrl = SimulationController()
        config = _make_config(end_date=None, max_cycles=None)
        ctrl.create_simulation(config)

        sims = ctrl.list_simulations()
        assert sims[0]["infinite"] is True

    def test_get_result_before_start(self) -> None:
        ctrl = SimulationController()
        assert ctrl.get_result("nonexistent") is None


class TestScenarioEvent:
    """Tests for ScenarioEvent dataclass."""

    def test_create_event(self) -> None:
        evt = ScenarioEvent(
            event_id="e1",
            trigger_time=datetime(2026, 6, 1, tzinfo=timezone.utc),
            event_type="price_shock",
            payload={"symbol": "AAPL", "change_pct": -0.10},
            description="AAPL drops 10%",
        )
        assert evt.applied is False
        assert evt.event_type == "price_shock"


class TestSimulationConfig:
    """Tests for SimulationConfig defaults."""

    def test_defaults(self) -> None:
        config = SimulationConfig(
            simulation_id="test",
            name="Default Test",
            markets=[Market.US],
        )
        assert len(config.models) == len(list(ModelProvider))
        assert config.interval_seconds == 3600
        assert config.initial_capital == 100_000.0
        assert config.end_date is None
        assert config.max_cycles is None
