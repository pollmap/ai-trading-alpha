"""ATLAS Benchmark Runner — main entry point for running the full benchmark.

Usage:
    python -m scripts.run_benchmark
    python -m scripts.run_benchmark --market CRYPTO
    python -m scripts.run_benchmark --cycles 10
"""

from __future__ import annotations

import argparse
import asyncio
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
)
from src.agents.orchestrator import BenchmarkOrchestrator
from src.data.normalizer import DataNormalizer
from src.data.scheduler import MarketScheduler
from src.llm.call_logger import LLMCallLogger
from src.llm.cost_tracker import CostTracker
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
    for market_key in ("KRX", "US", "CRYPTO"):
        section = raw.get(market_key, {})
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
}


class BenchmarkRunner:
    """High-level benchmark runner orchestrating all subsystems.

    Lifecycle:
        1. Initialize portfolios (9 per market)
        2. Start scheduler (fixed-interval + event triggers)
        3. Each cycle: collect data -> normalize -> run agents -> execute trades -> update PnL
        4. On shutdown: save final state + export reports
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

        # Subsystems
        self._cost_tracker = CostTracker()
        self._call_logger = LLMCallLogger()
        self._portfolio_manager = PortfolioManager()
        self._order_engine = OrderEngine()
        self._pnl_calculator = PnLCalculator()
        self._market_configs = _load_market_configs()
        self._normalizer = DataNormalizer()
        self._orchestrator = BenchmarkOrchestrator(
            cost_tracker=self._cost_tracker,
            call_logger=self._call_logger,
        )

    async def initialize(self) -> None:
        """Initialize all portfolios and register agents."""
        for market in self._markets:
            capital = INITIAL_CAPITALS.get(market, 100_000)
            self._portfolio_manager.init_portfolios(market, capital)
            log.info(
                "market_initialized",
                market=market.value,
                initial_capital=capital,
            )

        # Register agents (import adapters lazily to handle missing API keys)
        await self._register_agents()

        log.info(
            "benchmark_initialized",
            markets=[m.value for m in self._markets],
            agents=self._orchestrator.registered_agents,
        )

    async def _register_agents(self) -> None:
        """Attempt to register all agent combinations.

        Agents with missing API keys are skipped with a warning.
        """
        from config.settings import get_settings
        settings = get_settings()

        # Try each model adapter
        adapters: dict[ModelProvider, object | None] = {}

        # DeepSeek
        if settings.deepseek_api_key.get_secret_value():
            try:
                from src.llm.deepseek_adapter import DeepSeekAdapter
                adapters[ModelProvider.DEEPSEEK] = DeepSeekAdapter(
                    cost_tracker=self._cost_tracker,
                    call_logger=self._call_logger,
                )
            except Exception as exc:
                log.warning("deepseek_adapter_init_failed", error=str(exc))
        else:
            log.warning("deepseek_api_key_missing")

        # Gemini
        if settings.gemini_api_key.get_secret_value():
            try:
                from src.llm.gemini_adapter import GeminiAdapter
                adapters[ModelProvider.GEMINI] = GeminiAdapter(
                    cost_tracker=self._cost_tracker,
                    call_logger=self._call_logger,
                )
            except Exception as exc:
                log.warning("gemini_adapter_init_failed", error=str(exc))
        else:
            log.warning("gemini_api_key_missing")

        # Claude
        if settings.anthropic_api_key.get_secret_value():
            try:
                from src.llm.claude_adapter import ClaudeAdapter
                adapters[ModelProvider.CLAUDE] = ClaudeAdapter(
                    cost_tracker=self._cost_tracker,
                    call_logger=self._call_logger,
                )
            except Exception as exc:
                log.warning("claude_adapter_init_failed", error=str(exc))
        else:
            log.warning("anthropic_api_key_missing")

        # GPT
        if settings.openai_api_key.get_secret_value():
            try:
                from src.llm.gpt_adapter import GPTAdapter
                adapters[ModelProvider.GPT] = GPTAdapter(
                    cost_tracker=self._cost_tracker,
                    call_logger=self._call_logger,
                )
            except Exception as exc:
                log.warning("gpt_adapter_init_failed", error=str(exc))
        else:
            log.warning("openai_api_key_missing")

        # Register single agents for each available model
        from src.agents.single_agent import SingleAgent

        for model, adapter in adapters.items():
            if adapter is None:
                continue
            single_agent = SingleAgent(llm_adapter=adapter)
            self._orchestrator.register_agent(
                model=model,
                architecture=AgentArchitecture.SINGLE,
                agent=single_agent,
            )

        log.info(
            "agents_registered",
            count=len(self._orchestrator.registered_agents),
            models=[m.value for m in adapters if adapters[m] is not None],
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

                # Wait for next cycle (configurable per market, use shortest)
                wait_seconds = 15 * 60  # 15 min default (crypto)
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
        """Execute a single benchmark cycle."""
        cycle_start = datetime.now(timezone.utc)
        log.info("cycle_start", cycle=self._cycle_count + 1, timestamp=str(cycle_start))

        # 1. Collect and normalize market data
        snapshots = {}
        for market in self._markets:
            try:
                snapshot = await self._normalizer.build_snapshot(market)
                if snapshot is not None:
                    snapshots[market] = snapshot
            except Exception as exc:
                log.error("snapshot_failed", market=market.value, error=str(exc))

        if not snapshots:
            log.warning("no_snapshots_available")
            return

        # 2. Run all agents
        all_signals = await self._orchestrator.run_cycle(snapshots)

        # 3. Execute trades for each signal
        trade_count = 0
        for market_key, signals in all_signals.items():
            market = Market(market_key)
            for sig in signals:
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

                    market_cfg = self._market_configs.get(market.value, {})
                    trade = self._order_engine.execute_signal(
                        signal=sig,
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

                except Exception as exc:
                    log.error(
                        "trade_execution_failed",
                        signal_id=sig.signal_id,
                        error=str(exc),
                    )

        # 4. Update PnL for all portfolios
        for market in self._markets:
            snapshot = snapshots.get(market)
            if snapshot is None:
                continue
            current_prices = {
                sym: sd.close for sym, sd in snapshot.symbols.items()
            }
            for portfolio in self._portfolio_manager.snapshot_all():
                if portfolio.market == market:
                    PnLCalculator.update_portfolio_values(portfolio, current_prices)

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        log.info(
            "cycle_complete",
            cycle=self._cycle_count + 1,
            signals=sum(len(s) for s in all_signals.values()),
            trades=trade_count,
            elapsed_seconds=round(elapsed, 2),
            cost_summary=self._cost_tracker.summary(),
        )

    async def shutdown(self) -> None:
        """Graceful shutdown — save state and export reports."""
        self._running = False
        log.info(
            "benchmark_shutdown",
            total_cycles=self._cycle_count,
            total_cost=self._cost_tracker.summary(),
        )

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    @property
    def portfolio_manager(self) -> PortfolioManager:
        return self._portfolio_manager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ATLAS Benchmark Runner")
    parser.add_argument(
        "--market",
        type=str,
        nargs="+",
        default=["CRYPTO"],
        choices=["KRX", "US", "CRYPTO"],
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
