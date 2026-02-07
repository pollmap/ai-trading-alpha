"""Benchmark orchestrator — parallel execution of all agent combinations."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from src.core.logging import get_logger
from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    MarketSnapshot,
    ModelProvider,
    PortfolioState,
    TradingSignal,
)
from src.agents.single_agent import SingleAgent
from src.llm.call_logger import LLMCallLogger
from src.llm.cost_tracker import CostTracker

log = get_logger(__name__)

# All model-architecture combinations
AGENT_MATRIX: list[tuple[ModelProvider, AgentArchitecture]] = [
    (ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE),
    (ModelProvider.GEMINI, AgentArchitecture.SINGLE),
    (ModelProvider.CLAUDE, AgentArchitecture.SINGLE),
    (ModelProvider.GPT, AgentArchitecture.SINGLE),
    (ModelProvider.DEEPSEEK, AgentArchitecture.MULTI),
    (ModelProvider.GEMINI, AgentArchitecture.MULTI),
    (ModelProvider.CLAUDE, AgentArchitecture.MULTI),
    (ModelProvider.GPT, AgentArchitecture.MULTI),
]


class BenchmarkOrchestrator:
    """Orchestrate parallel execution of all agent combinations.

    Execution matrix: 4 models x 2 architectures + Buy&Hold = 9 tasks.

    Error isolation:
    - 1 agent failure -> only that agent gets HOLD, rest continue
    - DB save failure -> local fallback + warning
    - Full cycle failure -> 60s wait then retry
    """

    def __init__(
        self,
        cost_tracker: CostTracker,
        call_logger: LLMCallLogger,
        single_timeout: int = 30,
        multi_timeout: int = 120,
    ) -> None:
        self._cost_tracker = cost_tracker
        self._call_logger = call_logger
        self._single_timeout = single_timeout
        self._multi_timeout = multi_timeout
        self._agents: dict[tuple[str, str], SingleAgent] = {}
        self._cycle_count = 0

    def register_agent(
        self,
        model: ModelProvider,
        architecture: AgentArchitecture,
        agent: SingleAgent,
    ) -> None:
        """Register an agent for a model-architecture combination."""
        key = (model.value, architecture.value)
        self._agents[key] = agent
        log.info("agent_registered", model=model.value, architecture=architecture.value)

    async def run_cycle(
        self,
        snapshots: dict[Market, MarketSnapshot],
    ) -> dict[str, list[TradingSignal]]:
        """Run one complete benchmark cycle across all agents.

        Args:
            snapshots: Market -> MarketSnapshot for each active market.

        Returns:
            Dict of market -> list of signals from all agents.
        """
        self._cycle_count += 1
        log.info(
            "cycle_starting",
            cycle=self._cycle_count,
            markets=[m.value for m in snapshots],
        )

        all_signals: dict[str, list[TradingSignal]] = {}

        for market, snapshot in snapshots.items():
            signals = await self._run_market_cycle(market, snapshot)
            all_signals[market.value] = signals

        log.info(
            "cycle_complete",
            cycle=self._cycle_count,
            total_signals=sum(len(s) for s in all_signals.values()),
            cost_summary=self._cost_tracker.summary(),
        )

        return all_signals

    async def _run_market_cycle(
        self,
        market: Market,
        snapshot: MarketSnapshot,
    ) -> list[TradingSignal]:
        """Run all agent combinations for a single market."""
        tasks: list[asyncio.Task[TradingSignal]] = []

        for model, arch in AGENT_MATRIX:
            key = (model.value, arch.value)
            agent = self._agents.get(key)

            if agent is None:
                log.debug("agent_not_registered", model=model.value, architecture=arch.value)
                continue

            # Create a dummy portfolio for now — in production this comes from PortfolioManager
            portfolio = PortfolioState(
                portfolio_id=f"{model.value}_{arch.value}_{market.value}",
                model=model,
                architecture=arch,
                market=market,
                cash=100_000.0,
                positions={},
                initial_capital=100_000.0,
            )

            timeout = (
                self._single_timeout
                if arch == AgentArchitecture.SINGLE
                else self._multi_timeout
            )

            task = asyncio.create_task(
                self._run_agent_safe(agent, snapshot, portfolio, model, arch, timeout),
                name=f"{model.value}_{arch.value}",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: list[TradingSignal] = []
        for result in results:
            if isinstance(result, TradingSignal):
                signals.append(result)
            elif isinstance(result, Exception):
                log.error("agent_returned_exception", error=str(result))

        return signals

    async def _run_agent_safe(
        self,
        agent: SingleAgent,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
        model: ModelProvider,
        architecture: AgentArchitecture,
        timeout: int,
    ) -> TradingSignal:
        """Run a single agent with timeout and error isolation."""
        try:
            signal = await asyncio.wait_for(
                agent.run(snapshot, portfolio),
                timeout=timeout,
            )
            return signal

        except asyncio.TimeoutError:
            log.warning(
                "agent_timeout",
                model=model.value,
                architecture=architecture.value,
                timeout=timeout,
            )
            return self._hold_fallback(snapshot, model, architecture)

        except Exception as exc:
            log.error(
                "agent_execution_error",
                model=model.value,
                architecture=architecture.value,
                error=str(exc),
            )
            return self._hold_fallback(snapshot, model, architecture)

    def _hold_fallback(
        self,
        snapshot: MarketSnapshot,
        model: ModelProvider,
        architecture: AgentArchitecture,
    ) -> TradingSignal:
        """Create HOLD signal for failed agents."""
        from uuid_extensions import uuid7

        symbol = next(iter(snapshot.symbols), "UNKNOWN")
        return TradingSignal(
            signal_id=str(uuid7()),
            snapshot_id=snapshot.snapshot_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            market=snapshot.market,
            action=Action.HOLD,
            weight=0.0,
            confidence=0.0,
            reasoning="Agent execution failed — defaulting to HOLD",
            model=model,
            architecture=architecture,
        )

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def registered_agents(self) -> list[str]:
        return [f"{m}/{a}" for m, a in self._agents]
