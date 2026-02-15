"""ATLAS Benchmark Runner — main entry point for running the full benchmark.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --market CRYPTO
    python scripts/run_benchmark.py --market US CRYPTO --cycles 10
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import get_logger, setup_logging
from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    ModelProvider,
    PortfolioState,
)
from src.agents.orchestrator import BenchmarkOrchestrator
from src.analytics.results_store import ResultsStore
from src.data.collector import DataCollector
from src.llm.call_logger import LLMCallLogger
from src.llm.cost_tracker import CostTracker
from src.rl.gpu_position_sizer import GPUPositionSizer
from src.rl.position_sizer import SCALE_ACTIONS
from src.simulator.baselines import BuyAndHoldBaseline
from src.simulator.order_engine import OrderEngine
from src.simulator.pnl_calculator import PnLCalculator
from src.simulator.portfolio import PortfolioManager

# Market configs loaded from markets.yaml for order engine
_MARKET_CONFIGS: dict[str, dict[str, object]] = {}


def _load_market_configs() -> dict[str, dict[str, object]]:
    """Load market configs from markets.yaml for trade execution."""
    import yaml
    markets_yaml = Path(__file__).resolve().parent.parent / "config" / "markets.yaml"
    with markets_yaml.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    configs: dict[str, dict[str, object]] = {}
    for market_key, section in raw.items():
        if not isinstance(section, dict):
            continue
        configs[market_key] = {
            "commission": section.get("commission", {}),
            "slippage": section.get("slippage", 0.001),
            "max_position_weight": section.get("max_position_weight", 0.30),
            "min_cash_ratio": section.get("min_cash_ratio", 0.20),
        }
    return configs


setup_logging()
log = get_logger(__name__)

# ── Market initial capitals (from markets.yaml) ─────────────────
INITIAL_CAPITALS: dict[Market, float] = {
    Market.KRX: 100_000_000,  # 1억원
    Market.US: 100_000,
    Market.CRYPTO: 100_000,
    Market.JPX: 10_000_000,   # 1000万円
    Market.SSE: 500_000,      # 50万元
    Market.HKEX: 500_000,     # HK$50万
    Market.EURONEXT: 100_000,
    Market.LSE: 100_000,
    Market.BOND: 100_000,
    Market.COMMODITIES: 100_000,
}


class BenchmarkRunner:
    """High-level benchmark runner orchestrating all subsystems.

    Lifecycle:
        1. Initialize portfolios (9 per market)
        2. Collect market data via DataCollector
        3. Each cycle: collect data -> run agents -> execute trades -> update PnL
        4. Save results to JSONL via ResultsStore
        5. On shutdown: save final state
    """

    def __init__(
        self,
        markets: list[Market] | None = None,
        max_cycles: int = 0,
    ) -> None:
        self._markets = markets or [Market.CRYPTO]  # MVP: start with crypto
        self._max_cycles = max_cycles  # 0 = unlimited
        self._cycle_count = 0
        self._running = False
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._saved_cost_count = 0  # Track how many cost records have been saved

        # Subsystems
        self._cost_tracker = CostTracker()
        self._call_logger = LLMCallLogger()
        self._portfolio_manager = PortfolioManager()
        self._order_engine = OrderEngine()
        self._market_configs = _load_market_configs()
        self._data_collector = DataCollector()
        self._results_store = ResultsStore()
        self._orchestrator = BenchmarkOrchestrator(
            cost_tracker=self._cost_tracker,
            call_logger=self._call_logger,
            portfolio_manager=self._portfolio_manager,
        )

        # RL position sizer (GPU DQN with CPU Q-table fallback)
        self._position_sizer = GPUPositionSizer()
        self._rl_model_path = Path("data/rl_model.json")

        # Buy & Hold baseline per market
        self._baselines: dict[Market, BuyAndHoldBaseline] = {}

    async def initialize(self) -> None:
        """Initialize all portfolios and register agents."""
        for market in self._markets:
            capital = INITIAL_CAPITALS.get(market, 100_000)
            self._portfolio_manager.init_portfolios(market, capital)
            self._baselines[market] = BuyAndHoldBaseline()
            log.info(
                "market_initialized",
                market=market.value,
                initial_capital=capital,
            )

        # Load saved RL model if exists
        self._position_sizer.load(self._rl_model_path)

        # Try to restore from a previous crash
        if self._results_store.has_data():
            if self._try_restore():
                log.info("resumed_from_crash", cycle_count=self._cycle_count)

        # Register agents (import adapters lazily to handle missing API keys)
        await self._register_agents()

        log.info(
            "benchmark_initialized",
            markets=[m.value for m in self._markets],
            agents=self._orchestrator.registered_agents,
        )

        # Save initial status
        self._results_store.update_status(
            running=True,
            cycle_count=0,
            markets=[m.value for m in self._markets],
            registered_agents=self._orchestrator.registered_agents,
            total_cost_usd=0.0,
            started_at=self._started_at,
        )

        # Save initial portfolio snapshots (cycle 0)
        for portfolio in self._portfolio_manager.snapshot_all():
            self._results_store.save_portfolio_snapshot(portfolio, cycle=0)

    def _try_restore(self) -> bool:
        """Attempt to restore portfolio state from a previous run.

        Reads the last equity curve records and status.json to resume
        the benchmark from where it left off after a crash.

        Portfolios are matched by (model, architecture, market) tuple
        rather than portfolio_id, since init_portfolios() generates new
        UUIDs on every startup.

        Returns:
            True if state was successfully restored.
        """
        status = self._results_store.load_status()
        if status is None or not status.get("running", False):
            return False

        curves = self._results_store.load_equity_curves()
        if not curves:
            return False

        last_cycle = max(r["cycle"] for r in curves)
        self._cycle_count = last_cycle

        # Restore each portfolio's cash from the last equity curve record.
        # Use a seen-set to skip duplicate (model, arch, market) keys —
        # Buy & Hold portfolios share (DEEPSEEK, SINGLE, market) with the
        # real agent, so we keep only the first (agent) record per key.
        last_records = [r for r in curves if r["cycle"] == last_cycle]
        seen: set[tuple[str, str, str]] = set()
        restored = 0
        for record in last_records:
            try:
                model = ModelProvider(record["model"])
                arch = AgentArchitecture(record["architecture"])
                market = Market(record["market"])

                key = (record["model"], record["architecture"], record["market"])
                if key in seen:
                    continue  # skip B&H duplicate that shares the same enum combo
                seen.add(key)

                # Match by (model, arch, market) — works across restarts
                portfolio = self._portfolio_manager.get_state(model, arch, market)

                updated = PortfolioState(
                    portfolio_id=portfolio.portfolio_id,
                    model=model,
                    architecture=arch,
                    market=market,
                    cash=record.get("cash", portfolio.cash),
                    positions={},  # positions rebuilt from next market data fetch
                    initial_capital=record.get("initial_capital", portfolio.initial_capital),
                    created_at=portfolio.created_at,
                )
                self._portfolio_manager.set_state(portfolio.portfolio_id, updated)
                restored += 1
            except (KeyError, ValueError):
                # KeyError: portfolio not found; ValueError: invalid enum
                continue

        log.info("state_restored", cycle=last_cycle, portfolios=restored)
        return restored > 0

    async def _register_agents(self) -> None:
        """Attempt to register all agent combinations.

        Agents with missing API keys are skipped with a warning.
        For SINGLE architecture: uses SingleAgent.
        For MULTI architecture: uses MultiAgentPipeline (5-stage).
        """
        from config.settings import get_settings
        settings = get_settings()

        from src.agents.single_agent import SingleAgent

        adapter_configs: list[tuple[ModelProvider, str, str, str]] = [
            (ModelProvider.DEEPSEEK, "deepseek_api_key", "src.llm.deepseek_adapter", "DeepSeekAdapter"),
            (ModelProvider.GEMINI, "gemini_api_key", "src.llm.gemini_adapter", "GeminiAdapter"),
            (ModelProvider.CLAUDE, "anthropic_api_key", "src.llm.claude_adapter", "ClaudeAdapter"),
            (ModelProvider.GPT, "openai_api_key", "src.llm.gpt_adapter", "GPTAdapter"),
        ]

        for model, key_attr, module_path, class_name in adapter_configs:
            api_key = getattr(settings, key_attr).get_secret_value()
            if not api_key:
                log.warning(f"{key_attr}_missing", model=model.value)
                continue

            for arch in AgentArchitecture:
                try:
                    import importlib
                    mod = importlib.import_module(module_path)
                    adapter_cls = getattr(mod, class_name)
                    adapter = adapter_cls(
                        architecture=arch,
                        cost_tracker=self._cost_tracker,
                        call_logger=self._call_logger,
                    )

                    if arch == AgentArchitecture.MULTI:
                        # Use MultiAgentPipeline for MULTI architecture
                        from src.agents.multi_agent.graph import MultiAgentPipeline
                        agent: object = MultiAgentPipeline(
                            llm_adapter=adapter,
                            model_provider=model,
                        )
                    else:
                        agent = SingleAgent(llm_adapter=adapter)

                    self._orchestrator.register_agent(
                        model=model,
                        architecture=arch,
                        agent=agent,
                    )
                except Exception as exc:
                    log.warning(
                        "agent_init_failed",
                        model=model.value,
                        architecture=arch.value,
                        error=str(exc),
                    )

        log.info(
            "agents_registered",
            count=len(self._orchestrator.registered_agents),
            agents=self._orchestrator.registered_agents,
        )

    async def run(self) -> None:
        """Run the benchmark loop."""
        self._running = True

        log.info(
            "benchmark_starting",
            markets=[m.value for m in self._markets],
            max_cycles=self._max_cycles or "unlimited",
        )

        while self._running:
            try:
                await self._run_cycle()
                self._cycle_count += 1

                if self._max_cycles > 0 and self._cycle_count >= self._max_cycles:
                    log.info("benchmark_max_cycles_reached", cycles=self._cycle_count)
                    break

                # Wait for next cycle (use shortest interval from active markets)
                wait_seconds = 15 * 60  # 15 min default
                log.info("cycle_waiting", next_in_seconds=wait_seconds)
                await asyncio.sleep(wait_seconds)

            except asyncio.CancelledError:
                log.info("benchmark_cancelled")
                break
            except Exception as exc:
                log.error("cycle_error", error=str(exc), cycle=self._cycle_count)
                await asyncio.sleep(60)  # Wait 60s on error then retry

        await self.shutdown()

    async def _run_cycle(self) -> None:
        """Execute a single benchmark cycle.

        Pipeline: collect data -> run agents -> execute baselines ->
                  execute trades -> update PnL -> save results
        """
        cycle_start = datetime.now(timezone.utc)
        cycle_num = self._cycle_count + 1
        log.info("cycle_start", cycle=cycle_num, timestamp=str(cycle_start))

        # 1. Collect market data via DataCollector
        snapshots = await self._data_collector.collect_all(self._markets)

        if not snapshots:
            log.warning("no_snapshots_available", cycle=cycle_num)
            return

        log.info(
            "data_collected",
            cycle=cycle_num,
            markets=[m.value for m in snapshots],
            symbols={m.value: len(s.symbols) for m, s in snapshots.items()},
        )

        # 2. Run all LLM agents via orchestrator
        all_signals = await self._orchestrator.run_cycle(snapshots)

        trade_count = 0

        # 3. Run Buy & Hold baseline for each market (execute separately)
        for market, snapshot in snapshots.items():
            baseline = self._baselines.get(market)
            if baseline is None:
                continue
            try:
                bh_portfolio = self._portfolio_manager.get_buy_hold_state(market)
                bh_signals = baseline.generate_signals(snapshot, bh_portfolio)

                for sig in bh_signals:
                    self._results_store.save_signal(sig, cycle=cycle_num)

                    symbol_data = snapshot.symbols.get(sig.symbol)
                    if symbol_data is None or sig.action == Action.HOLD:
                        continue

                    market_cfg = self._market_configs.get(market.value, {})
                    trade = self._order_engine.execute_signal(
                        signal=sig,
                        portfolio=bh_portfolio,
                        market_config=market_cfg,
                        close_price=symbol_data.close,
                    )

                    if trade is not None:
                        qty_delta = trade.quantity if trade.action == Action.BUY else -trade.quantity
                        self._portfolio_manager.update_position(
                            bh_portfolio.portfolio_id,
                            trade.symbol,
                            qty_delta,
                            trade.price,
                            trade.commission,
                        )
                        # Refresh portfolio state after each trade
                        bh_portfolio = self._portfolio_manager.get_buy_hold_state(market)
                        trade_count += 1

                        self._results_store.save_trade(
                            trade_id=trade.trade_id,
                            signal_id=trade.signal_id,
                            symbol=trade.symbol,
                            action=trade.action.value,
                            quantity=trade.quantity,
                            price=trade.price,
                            commission=trade.commission,
                            realized_pnl=trade.realized_pnl,
                            cycle=cycle_num,
                        )

                log.info(
                    "baseline_signals_generated",
                    market=market.value,
                    n_signals=len(bh_signals),
                    cycle=cycle_num,
                )
            except Exception as exc:
                log.error(
                    "baseline_execution_failed",
                    market=market.value,
                    error=str(exc),
                )

        # 4. Execute trades for each LLM signal + save signals
        for market_key, signals in all_signals.items():
            market = Market(market_key)
            for sig in signals:
                # Save every signal
                self._results_store.save_signal(sig, cycle=cycle_num)

                try:
                    portfolio = self._portfolio_manager.get_state(
                        sig.model, sig.architecture, market,
                    )

                    # Get current price from snapshot
                    snapshot = snapshots.get(market)
                    if snapshot is None:
                        continue
                    symbol_data = snapshot.symbols.get(sig.symbol)
                    if symbol_data is None:
                        continue

                    # RL position sizer adjusts signal weight
                    sizing = self._position_sizer.decide(
                        signal=sig,
                        portfolio=portfolio,
                    )
                    adjusted_sig = dataclasses.replace(sig, weight=sizing.scaled_weight)

                    market_cfg = self._market_configs.get(market.value, {})
                    trade = self._order_engine.execute_signal(
                        signal=adjusted_sig,
                        portfolio=portfolio,
                        market_config=market_cfg,
                        close_price=symbol_data.close,
                    )

                    if trade is not None:
                        # Apply the trade to portfolio
                        qty_delta = trade.quantity if trade.action == Action.BUY else -trade.quantity
                        self._portfolio_manager.update_position(
                            portfolio.portfolio_id,
                            trade.symbol,
                            qty_delta,
                            trade.price,
                            trade.commission,
                        )
                        trade_count += 1

                        # RL update: reward = normalized realized PnL
                        if portfolio.initial_capital > 0:
                            reward = trade.realized_pnl / portfolio.initial_capital
                        else:
                            reward = 0.0
                        updated_portfolio = self._portfolio_manager.get_state(
                            sig.model, sig.architecture, market,
                        )
                        next_sizing = self._position_sizer.decide(
                            signal=sig,
                            portfolio=updated_portfolio,
                        )
                        self._position_sizer.update(
                            state_key=sizing.state.key,
                            action_idx=SCALE_ACTIONS.index(sizing.scale_factor),
                            reward=reward,
                            next_state_key=next_sizing.state.key,
                        )

                        # Save trade
                        self._results_store.save_trade(
                            trade_id=trade.trade_id,
                            signal_id=trade.signal_id,
                            symbol=trade.symbol,
                            action=trade.action.value,
                            quantity=trade.quantity,
                            price=trade.price,
                            commission=trade.commission,
                            realized_pnl=trade.realized_pnl,
                            cycle=cycle_num,
                        )

                except Exception as exc:
                    log.error(
                        "trade_execution_failed",
                        signal_id=sig.signal_id,
                        error=str(exc),
                    )

        # 5. Update PnL mark-to-market for all portfolios
        for market in self._markets:
            snapshot = snapshots.get(market)
            if snapshot is None:
                continue
            current_prices = {
                sym: sd.close for sym, sd in snapshot.symbols.items()
            }
            for portfolio in self._portfolio_manager.snapshot_all():
                if portfolio.market == market:
                    updated = PnLCalculator.update_portfolio_values(portfolio, current_prices)
                    self._portfolio_manager.set_state(portfolio.portfolio_id, updated)

        # 6. Save portfolio snapshots + cost records
        for portfolio in self._portfolio_manager.snapshot_all():
            self._results_store.save_portfolio_snapshot(portfolio, cycle=cycle_num)

        # Save new cost records from this cycle
        all_cost_records = self._cost_tracker.get_records()
        new_records = all_cost_records[self._saved_cost_count:]
        self._saved_cost_count = len(all_cost_records)
        for record in new_records:
            self._results_store.save_cost_record(
                model=record.model,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cost_usd=record.cost_usd,
                latency_ms=record.latency_ms,
                cycle=cycle_num,
            )

        # 7. Update status
        self._results_store.update_status(
            running=self._running,
            cycle_count=cycle_num,
            markets=[m.value for m in self._markets],
            registered_agents=self._orchestrator.registered_agents,
            total_cost_usd=self._cost_tracker.total_cost_usd,
            started_at=self._started_at,
        )

        # 8. Save RL model periodically
        self._position_sizer.save(self._rl_model_path)

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        log.info(
            "cycle_complete",
            cycle=cycle_num,
            signals=sum(len(s) for s in all_signals.values()),
            trades=trade_count,
            elapsed_seconds=round(elapsed, 2),
            cost_summary=self._cost_tracker.summary(),
            rl_steps=self._position_sizer.total_steps,
        )

    async def shutdown(self) -> None:
        """Graceful shutdown — save final state."""
        self._running = False

        # Save final status
        self._results_store.update_status(
            running=False,
            cycle_count=self._cycle_count,
            markets=[m.value for m in self._markets],
            registered_agents=self._orchestrator.registered_agents,
            total_cost_usd=self._cost_tracker.total_cost_usd,
            started_at=self._started_at,
        )

        log.info(
            "benchmark_shutdown",
            total_cycles=self._cycle_count,
            total_cost=self._cost_tracker.summary(),
            results_dir=str(self._results_store.results_dir),
        )

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    @property
    def portfolio_manager(self) -> PortfolioManager:
        return self._portfolio_manager

    @property
    def results_store(self) -> ResultsStore:
        return self._results_store


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ATLAS Benchmark Runner")
    parser.add_argument(
        "--market",
        type=str,
        nargs="+",
        default=["CRYPTO"],
        choices=[m.value for m in Market],
        help="Markets to benchmark (default: CRYPTO)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Max cycles (0 = unlimited, default: 0)",
    )
    return parser.parse_args()


async def main() -> None:
    """Entry point."""
    args = parse_args()
    markets = [Market(m) for m in args.market]

    runner = BenchmarkRunner(markets=markets, max_cycles=args.cycles)

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    for sig_name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig_name,
            lambda: asyncio.create_task(runner.shutdown()),
        )

    await runner.initialize()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
