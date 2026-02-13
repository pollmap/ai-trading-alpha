"""End-to-end integration test — full benchmark cycle with mock LLM.

Proves the complete pipeline works without real API keys:
    init portfolios -> collect data (mock) -> run agents (mock LLM) ->
    execute trades -> update PnL -> save results -> verify JSONL output
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from uuid_extensions import uuid7

from src.agents.orchestrator import BenchmarkOrchestrator
from src.agents.single_agent import SingleAgent
from src.analytics.results_store import ResultsStore
from src.core.interfaces import BaseLLMAdapter
from src.core.types import (
    Action,
    AgentArchitecture,
    MacroData,
    Market,
    MarketSnapshot,
    ModelProvider,
    PortfolioState,
    SymbolData,
    TradingSignal,
)
from src.llm.call_logger import LLMCallLogger
from src.llm.cost_tracker import CostTracker
from src.simulator.baselines import BuyAndHoldBaseline
from src.simulator.order_engine import OrderEngine
from src.simulator.pnl_calculator import PnLCalculator
from src.simulator.portfolio import PortfolioManager


# -- Mock LLM Adapter --


class MockLLMAdapter(BaseLLMAdapter):
    """Deterministic LLM adapter that returns predictable BUY signals."""

    def __init__(
        self,
        model: ModelProvider,
        architecture: AgentArchitecture,
        action: Action = Action.BUY,
    ) -> None:
        self._model = model
        self._architecture = architecture
        self._action = action
        self._call_count = 0

    async def generate_signal(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TradingSignal:
        self._call_count += 1
        symbol = next(iter(snapshot.symbols))
        return TradingSignal(
            signal_id=str(uuid7()),
            snapshot_id=snapshot.snapshot_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            market=snapshot.market,
            action=self._action,
            weight=0.25,
            confidence=0.85,
            reasoning=f"Mock {self._model.value}/{self._architecture.value} signal: {self._action.value} {symbol}",
            model=self._model,
            architecture=self._architecture,
            latency_ms=50.0,
        )


# -- Fixtures --


@pytest.fixture
def snapshot() -> MarketSnapshot:
    """Create a realistic CRYPTO market snapshot."""
    return MarketSnapshot(
        snapshot_id=str(uuid7()),
        timestamp=datetime.now(timezone.utc),
        market=Market.CRYPTO,
        symbols={
            "BTCUSDT": SymbolData(
                symbol="BTCUSDT",
                market=Market.CRYPTO,
                open=50_000.0,
                high=51_000.0,
                low=49_000.0,
                close=50_500.0,
                volume=1_000_000.0,
                currency="USDT",
            ),
            "ETHUSDT": SymbolData(
                symbol="ETHUSDT",
                market=Market.CRYPTO,
                open=3_000.0,
                high=3_100.0,
                low=2_900.0,
                close=3_050.0,
                volume=500_000.0,
                currency="USDT",
            ),
        },
        macro=MacroData(vix=18.5, fear_greed_index=65.0),
    )


@pytest.fixture
def results_dir():
    """Create a temporary results directory."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# -- Tests --


class TestFullPipelineE2E:
    """End-to-end test of the complete benchmark pipeline."""

    def test_full_cycle_with_mock_llm(
        self, snapshot: MarketSnapshot, results_dir: Path,
    ) -> None:
        """Run a full cycle and verify everything works end-to-end."""
        import asyncio

        # 1. Initialize subsystems
        portfolio_mgr = PortfolioManager()
        portfolio_mgr.init_portfolios(Market.CRYPTO, initial_capital=100_000.0)

        cost_tracker = CostTracker()
        call_logger = LLMCallLogger()
        order_engine = OrderEngine()
        results_store = ResultsStore(results_dir)
        baseline = BuyAndHoldBaseline()

        orchestrator = BenchmarkOrchestrator(
            cost_tracker=cost_tracker,
            call_logger=call_logger,
            portfolio_manager=portfolio_mgr,
        )

        # Register mock agents (1 model, both architectures)
        mock_single = MockLLMAdapter(ModelProvider.GPT, AgentArchitecture.SINGLE, Action.BUY)
        mock_multi = MockLLMAdapter(ModelProvider.GPT, AgentArchitecture.MULTI, Action.BUY)

        orchestrator.register_agent(
            ModelProvider.GPT,
            AgentArchitecture.SINGLE,
            SingleAgent(llm_adapter=mock_single),
        )
        orchestrator.register_agent(
            ModelProvider.GPT,
            AgentArchitecture.MULTI,
            SingleAgent(llm_adapter=mock_multi),
        )

        assert len(orchestrator.registered_agents) == 2

        # 2. Save initial portfolio snapshots (cycle 0)
        for portfolio in portfolio_mgr.snapshot_all():
            results_store.save_portfolio_snapshot(portfolio, cycle=0)

        # 3. Run cycle 1
        loop = asyncio.new_event_loop()

        snapshots = {Market.CRYPTO: snapshot}
        all_signals = loop.run_until_complete(orchestrator.run_cycle(snapshots))

        # Verify signals were generated
        assert "CRYPTO" in all_signals
        llm_signals = all_signals["CRYPTO"]
        assert len(llm_signals) >= 2

        # Run Buy & Hold baseline — execute against B&H portfolio directly
        bh_portfolio = portfolio_mgr.get_buy_hold_state(Market.CRYPTO)
        bh_signals = baseline.generate_signals(snapshot, bh_portfolio)
        assert len(bh_signals) == 2  # BUY for BTCUSDT and ETHUSDT
        assert all(s.action == Action.BUY for s in bh_signals)

        # Execute B&H trades against B&H portfolio
        for sig in bh_signals:
            results_store.save_signal(sig, cycle=1)
            symbol_data = snapshot.symbols.get(sig.symbol)
            if symbol_data is None or sig.action == Action.HOLD:
                continue
            trade = order_engine.execute_signal(
                signal=sig, portfolio=bh_portfolio,
                market_config={}, close_price=symbol_data.close,
            )
            if trade is not None:
                qty_delta = trade.quantity if trade.action == Action.BUY else -trade.quantity
                portfolio_mgr.update_position(
                    bh_portfolio.portfolio_id, trade.symbol,
                    qty_delta, trade.price, trade.commission,
                )
                bh_portfolio = portfolio_mgr.get_buy_hold_state(Market.CRYPTO)

        # 4. Execute trades for LLM signals
        trade_count = 0
        for sig in all_signals["CRYPTO"]:
            results_store.save_signal(sig, cycle=1)

            try:
                portfolio = portfolio_mgr.get_state(
                    sig.model, sig.architecture, Market.CRYPTO,
                )
                symbol_data = snapshot.symbols.get(sig.symbol)
                if symbol_data is None:
                    continue

                trade = order_engine.execute_signal(
                    signal=sig,
                    portfolio=portfolio,
                    market_config={},
                    close_price=symbol_data.close,
                )

                if trade is not None:
                    qty_delta = trade.quantity if trade.action == Action.BUY else -trade.quantity
                    portfolio_mgr.update_position(
                        portfolio.portfolio_id,
                        trade.symbol,
                        qty_delta,
                        trade.price,
                        trade.commission,
                    )
                    trade_count += 1

                    results_store.save_trade(
                        trade_id=trade.trade_id,
                        signal_id=trade.signal_id,
                        symbol=trade.symbol,
                        action=trade.action.value,
                        quantity=trade.quantity,
                        price=trade.price,
                        commission=trade.commission,
                        realized_pnl=trade.realized_pnl,
                        cycle=1,
                    )
            except Exception:
                pass  # Some signals may fail (e.g. insufficient funds)

        assert trade_count >= 1, "At least one trade should have executed"

        # 5. Update PnL mark-to-market
        current_prices = {
            sym: sd.close for sym, sd in snapshot.symbols.items()
        }
        for portfolio in portfolio_mgr.snapshot_all():
            if portfolio.market == Market.CRYPTO:
                updated = PnLCalculator.update_portfolio_values(portfolio, current_prices)
                portfolio_mgr.set_state(portfolio.portfolio_id, updated)

        # 6. Save portfolio snapshots
        for portfolio in portfolio_mgr.snapshot_all():
            results_store.save_portfolio_snapshot(portfolio, cycle=1)

        # 7. Save status
        results_store.update_status(
            running=True,
            cycle_count=1,
            markets=["CRYPTO"],
            registered_agents=orchestrator.registered_agents,
            total_cost_usd=0.0,
        )

        # -- Verify persisted results --

        saved_signals = results_store.load_signals()
        assert len(saved_signals) >= 4  # 2 LLM + 2 B&H signals

        saved_trades = results_store.load_trades()
        assert len(saved_trades) >= 1

        saved_curves = results_store.load_equity_curves()
        assert len(saved_curves) >= 18  # 9 portfolios x 2 cycles

        status = results_store.load_status()
        assert status is not None
        assert status["cycle_count"] == 1
        assert status["running"] is True
        assert "CRYPTO" in status["markets"]

        assert results_store.has_data()

        # At least one portfolio has positions
        all_portfolios = portfolio_mgr.snapshot_all()
        portfolios_with_positions = [p for p in all_portfolios if len(p.positions) > 0]
        assert len(portfolios_with_positions) >= 1

        # Buy & Hold portfolio has spent some cash
        bh_state = portfolio_mgr.get_buy_hold_state(Market.CRYPTO)
        assert bh_state.cash < 100_000.0, "B&H should have spent some cash"

        loop.close()

    def test_buy_hold_cycle2_emits_hold(
        self, snapshot: MarketSnapshot,
    ) -> None:
        """After initial buy, Buy & Hold should emit HOLD signals."""
        portfolio_mgr = PortfolioManager()
        portfolio_mgr.init_portfolios(Market.CRYPTO, initial_capital=100_000.0)
        bh_portfolio = portfolio_mgr.get_buy_hold_state(Market.CRYPTO)

        baseline = BuyAndHoldBaseline()

        # Cycle 1: BUY
        signals_1 = baseline.generate_signals(snapshot, bh_portfolio)
        assert all(s.action == Action.BUY for s in signals_1)

        # Cycle 2: HOLD
        signals_2 = baseline.generate_signals(snapshot, bh_portfolio)
        assert all(s.action == Action.HOLD for s in signals_2)


class TestPortfolioEncapsulation:
    """Verify set_state() encapsulation fix."""

    def test_set_state_replaces_internal_state(self) -> None:
        """set_state() should replace the portfolio state."""
        mgr = PortfolioManager()
        mgr.init_portfolios(Market.US, initial_capital=50_000.0)

        state = mgr.get_state(ModelProvider.GPT, AgentArchitecture.SINGLE, Market.US)
        original_cash = state.cash

        modified = PortfolioState(
            portfolio_id=state.portfolio_id,
            model=state.model,
            architecture=state.architecture,
            market=state.market,
            cash=state.cash - 1000,
            positions=state.positions,
            initial_capital=state.initial_capital,
            created_at=state.created_at,
        )

        mgr.set_state(state.portfolio_id, modified)
        updated = mgr.get_state(ModelProvider.GPT, AgentArchitecture.SINGLE, Market.US)
        assert updated.cash == original_cash - 1000

    def test_set_state_rejects_unknown_id(self) -> None:
        """set_state() should raise KeyError for unknown portfolio_id."""
        mgr = PortfolioManager()
        dummy = PortfolioState(
            portfolio_id="nonexistent",
            model=ModelProvider.GPT,
            architecture=AgentArchitecture.SINGLE,
            market=Market.US,
            cash=100_000.0,
            positions={},
            initial_capital=100_000.0,
        )
        with pytest.raises(KeyError):
            mgr.set_state("nonexistent", dummy)


class TestOrchestratorAgentTypes:
    """Verify orchestrator accepts different agent types."""

    def test_accepts_both_architectures(
        self, snapshot: MarketSnapshot,
    ) -> None:
        """Orchestrator should produce signals for both SINGLE and MULTI agents."""
        import asyncio

        portfolio_mgr = PortfolioManager()
        portfolio_mgr.init_portfolios(Market.CRYPTO, initial_capital=100_000.0)

        cost_tracker = CostTracker()
        call_logger = LLMCallLogger()

        orchestrator = BenchmarkOrchestrator(
            cost_tracker=cost_tracker,
            call_logger=call_logger,
            portfolio_manager=portfolio_mgr,
        )

        mock_adapter_s = MockLLMAdapter(ModelProvider.CLAUDE, AgentArchitecture.SINGLE)
        mock_adapter_m = MockLLMAdapter(ModelProvider.CLAUDE, AgentArchitecture.MULTI)

        orchestrator.register_agent(
            ModelProvider.CLAUDE,
            AgentArchitecture.SINGLE,
            SingleAgent(llm_adapter=mock_adapter_s),
        )
        orchestrator.register_agent(
            ModelProvider.CLAUDE,
            AgentArchitecture.MULTI,
            SingleAgent(llm_adapter=mock_adapter_m),
        )

        loop = asyncio.new_event_loop()
        signals = loop.run_until_complete(
            orchestrator.run_cycle({Market.CRYPTO: snapshot}),
        )
        loop.close()

        assert "CRYPTO" in signals
        assert len(signals["CRYPTO"]) == 2
        archs = {s.architecture for s in signals["CRYPTO"]}
        assert AgentArchitecture.SINGLE in archs
        assert AgentArchitecture.MULTI in archs


class TestResultsStoreRoundTrip:
    """Verify JSONL read/write round-trip."""

    def test_signal_round_trip(self, results_dir: Path) -> None:
        """Signals should survive write + read cycle."""
        store = ResultsStore(results_dir)
        sig = TradingSignal(
            signal_id="sig-test-1",
            snapshot_id="snap-test-1",
            timestamp=datetime.now(timezone.utc),
            symbol="AAPL",
            market=Market.US,
            action=Action.BUY,
            weight=0.3,
            confidence=0.9,
            reasoning="Test signal for round-trip verification",
            model=ModelProvider.GPT,
            architecture=AgentArchitecture.SINGLE,
            latency_ms=100.0,
        )
        store.save_signal(sig, cycle=1)
        loaded = store.load_signals()
        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "AAPL"
        assert loaded[0]["action"] == "BUY"

    def test_status_round_trip(self, results_dir: Path) -> None:
        """Status should survive write + read cycle."""
        store = ResultsStore(results_dir)
        store.update_status(
            running=True,
            cycle_count=5,
            markets=["US", "CRYPTO"],
            registered_agents=["gpt/single", "claude/multi"],
            total_cost_usd=1.234,
        )
        status = store.load_status()
        assert status["cycle_count"] == 5
        assert status["total_cost_usd"] == 1.234
        assert "US" in status["markets"]
