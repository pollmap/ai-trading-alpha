"""Simulation controller -- manages simulation lifecycle with custom parameters.

Supports:
- Custom date ranges (start_date to end_date)
- Infinite mode (runs until manually stopped)
- Scenario injection (add events at any timing)
- Pause / Resume / Stop controls
- Per-simulation isolated state
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Awaitable

from src.core.logging import get_logger
from src.core.types import Market, ModelProvider, AgentArchitecture, MarketSnapshot, NewsItem

log = get_logger(__name__)


class SimulationStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ScenarioEvent:
    """Injectable scenario event."""

    event_id: str
    trigger_time: datetime  # When to inject (UTC)
    event_type: str  # "news" | "price_shock" | "macro_change" | "custom"
    payload: dict[str, object] = field(default_factory=dict)
    description: str = ""
    applied: bool = False


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    simulation_id: str
    name: str
    markets: list[Market]
    models: list[ModelProvider] = field(default_factory=lambda: list(ModelProvider))
    architectures: list[AgentArchitecture] = field(default_factory=lambda: list(AgentArchitecture))
    start_date: datetime | None = None  # None = now
    end_date: datetime | None = None    # None = infinite mode
    interval_seconds: int = 3600        # How often to run cycles (default 1 hour)
    initial_capital: float = 100_000.0
    scenarios: list[ScenarioEvent] = field(default_factory=list)
    max_cycles: int | None = None       # Safety limit for infinite mode
    enable_rl_sizer: bool = True
    custom_strategy_ids: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result summary after simulation completes."""

    simulation_id: str
    config: SimulationConfig
    status: SimulationStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_cycles: int = 0
    scenarios_applied: int = 0
    error_message: str | None = None
    portfolio_snapshots: list[dict[str, object]] = field(default_factory=list)


class SimulationController:
    """Manages simulation lifecycle."""

    def __init__(self) -> None:
        self._simulations: dict[str, SimulationConfig] = {}
        self._states: dict[str, SimulationStatus] = {}
        self._results: dict[str, SimulationResult] = {}
        self._pause_events: dict[str, asyncio.Event] = {}
        self._stop_flags: dict[str, bool] = {}
        self._cycle_counts: dict[str, int] = {}
        self._callbacks: dict[str, list[Callable[[str, int, dict[str, object]], Awaitable[None]]]] = {}

    def create_simulation(self, config: SimulationConfig) -> str:
        """Register a new simulation. Returns simulation_id."""
        sid = config.simulation_id
        self._simulations[sid] = config
        self._states[sid] = SimulationStatus.CREATED
        self._pause_events[sid] = asyncio.Event()
        self._pause_events[sid].set()  # Not paused initially
        self._stop_flags[sid] = False
        self._cycle_counts[sid] = 0
        self._callbacks[sid] = []

        log.info(
            "simulation_created",
            simulation_id=sid,
            name=config.name,
            markets=[m.value for m in config.markets],
            infinite=config.end_date is None,
        )
        return sid

    async def start(self, simulation_id: str) -> SimulationResult:
        """Start the simulation loop. Blocks until completion/stop."""
        config = self._simulations[simulation_id]
        self._states[simulation_id] = SimulationStatus.RUNNING

        result = SimulationResult(
            simulation_id=simulation_id,
            config=config,
            status=SimulationStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        self._results[simulation_id] = result

        log.info("simulation_started", simulation_id=simulation_id)

        try:
            current_time = config.start_date or datetime.now(timezone.utc)

            while True:
                # Check stop flag
                if self._stop_flags.get(simulation_id, False):
                    self._states[simulation_id] = SimulationStatus.STOPPED
                    result.status = SimulationStatus.STOPPED
                    break

                # Wait if paused
                await self._pause_events[simulation_id].wait()

                # Check end condition
                if config.end_date and current_time >= config.end_date:
                    self._states[simulation_id] = SimulationStatus.COMPLETED
                    result.status = SimulationStatus.COMPLETED
                    break

                # Check max cycles
                if config.max_cycles and self._cycle_counts[simulation_id] >= config.max_cycles:
                    self._states[simulation_id] = SimulationStatus.COMPLETED
                    result.status = SimulationStatus.COMPLETED
                    break

                # Apply pending scenarios
                scenarios_applied = self._apply_scenarios(simulation_id, current_time)
                result.scenarios_applied += scenarios_applied

                # Execute one cycle
                cycle_data: dict[str, object] = {
                    "cycle": self._cycle_counts[simulation_id],
                    "timestamp": current_time.isoformat(),
                    "markets": [m.value for m in config.markets],
                }

                # Notify callbacks
                for cb in self._callbacks.get(simulation_id, []):
                    await cb(simulation_id, self._cycle_counts[simulation_id], cycle_data)

                self._cycle_counts[simulation_id] += 1
                result.total_cycles = self._cycle_counts[simulation_id]

                log.debug(
                    "simulation_cycle",
                    simulation_id=simulation_id,
                    cycle=self._cycle_counts[simulation_id],
                    timestamp=current_time.isoformat(),
                )

                # Advance time
                current_time += timedelta(seconds=config.interval_seconds)

                # Small yield to event loop
                await asyncio.sleep(0)

        except Exception as exc:
            self._states[simulation_id] = SimulationStatus.ERROR
            result.status = SimulationStatus.ERROR
            result.error_message = str(exc)
            log.error("simulation_error", simulation_id=simulation_id, error=str(exc))

        result.completed_at = datetime.now(timezone.utc)
        self._results[simulation_id] = result
        log.info(
            "simulation_finished",
            simulation_id=simulation_id,
            status=result.status.value,
            cycles=result.total_cycles,
        )
        return result

    def pause(self, simulation_id: str) -> None:
        """Pause a running simulation."""
        if self._states.get(simulation_id) == SimulationStatus.RUNNING:
            self._pause_events[simulation_id].clear()
            self._states[simulation_id] = SimulationStatus.PAUSED
            log.info("simulation_paused", simulation_id=simulation_id)

    def resume(self, simulation_id: str) -> None:
        """Resume a paused simulation."""
        if self._states.get(simulation_id) == SimulationStatus.PAUSED:
            self._states[simulation_id] = SimulationStatus.RUNNING
            self._pause_events[simulation_id].set()
            log.info("simulation_resumed", simulation_id=simulation_id)

    def stop(self, simulation_id: str) -> None:
        """Stop a simulation."""
        self._stop_flags[simulation_id] = True
        # Also resume if paused, so the loop can exit
        if simulation_id in self._pause_events:
            self._pause_events[simulation_id].set()
        log.info("simulation_stop_requested", simulation_id=simulation_id)

    def inject_scenario(self, simulation_id: str, event: ScenarioEvent) -> None:
        """Inject a scenario event into a running simulation."""
        config = self._simulations.get(simulation_id)
        if config:
            config.scenarios.append(event)
            log.info(
                "scenario_injected",
                simulation_id=simulation_id,
                event_id=event.event_id,
                event_type=event.event_type,
            )

    def get_status(self, simulation_id: str) -> SimulationStatus:
        """Return the current status of a simulation."""
        return self._states.get(simulation_id, SimulationStatus.CREATED)

    def get_result(self, simulation_id: str) -> SimulationResult | None:
        """Return the result of a simulation, if available."""
        return self._results.get(simulation_id)

    def get_cycle_count(self, simulation_id: str) -> int:
        """Return the number of cycles completed for a simulation."""
        return self._cycle_counts.get(simulation_id, 0)

    def on_cycle(
        self,
        simulation_id: str,
        callback: Callable[[str, int, dict[str, object]], Awaitable[None]],
    ) -> None:
        """Register a callback for each cycle."""
        self._callbacks.setdefault(simulation_id, []).append(callback)

    def list_simulations(self) -> list[dict[str, object]]:
        """List all simulations with their status."""
        return [
            {
                "simulation_id": sid,
                "name": config.name,
                "status": self._states.get(sid, SimulationStatus.CREATED).value,
                "cycles": self._cycle_counts.get(sid, 0),
                "markets": [m.value for m in config.markets],
                "infinite": config.end_date is None,
            }
            for sid, config in self._simulations.items()
        ]

    def _apply_scenarios(self, simulation_id: str, current_time: datetime) -> int:
        """Apply any scenarios that should trigger at current_time."""
        config = self._simulations[simulation_id]
        applied = 0
        for event in config.scenarios:
            if not event.applied and event.trigger_time <= current_time:
                event.applied = True
                applied += 1
                log.info(
                    "scenario_applied",
                    simulation_id=simulation_id,
                    event_id=event.event_id,
                    event_type=event.event_type,
                    description=event.description,
                )
        return applied
