"""Integration test for the Multi-Agent pipeline.

Verifies that the 5-stage pipeline (analysts -> debate -> trader ->
risk manager -> fund manager) calls call_with_prompt() with
role-specific system prompts, not the generic generate_signal().
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from uuid_extensions import uuid7

from src.agents.multi_agent.graph import MultiAgentPipeline
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


class PromptCapturingAdapter(BaseLLMAdapter):
    """Mock adapter that captures system_prompt from call_with_prompt().

    Records each prompt so tests can verify that the multi-agent
    pipeline is sending role-specific prompts to the LLM.
    """

    def __init__(self) -> None:
        self.captured_system_prompts: list[str] = []
        self.captured_user_prompts: list[str] = []
        self._call_count = 0

    async def generate_signal(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TradingSignal:
        """Should NOT be called by the fixed multi-agent pipeline."""
        self._call_count += 1
        return self._make_signal(snapshot, "generate_signal called (BUG!)")

    async def call_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TradingSignal:
        """Capture the prompts and return a deterministic signal."""
        self.captured_system_prompts.append(system_prompt)
        self.captured_user_prompts.append(user_prompt)
        self._call_count += 1

        # Return approve-friendly signal so risk manager stage passes
        return self._make_signal(snapshot, "I approve this trade proposal.")

    def _make_signal(self, snapshot: MarketSnapshot, reasoning: str) -> TradingSignal:
        return TradingSignal(
            signal_id=str(uuid7()),
            snapshot_id=snapshot.snapshot_id,
            timestamp=datetime.now(timezone.utc),
            symbol=next(iter(snapshot.symbols), "UNKNOWN"),
            market=snapshot.market,
            action=Action.BUY,
            weight=0.25,
            confidence=0.8,
            reasoning=reasoning,
            model=ModelProvider.GPT,
            architecture=AgentArchitecture.MULTI,
        )


@pytest.fixture
def snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=str(uuid7()),
        timestamp=datetime.now(timezone.utc),
        market=Market.CRYPTO,
        symbols={
            "BTCUSDT": SymbolData(
                symbol="BTCUSDT", market=Market.CRYPTO,
                open=50_000.0, high=51_000.0, low=49_000.0, close=50_500.0,
                volume=1_000_000.0, currency="USDT",
            ),
            "ETHUSDT": SymbolData(
                symbol="ETHUSDT", market=Market.CRYPTO,
                open=3_000.0, high=3_100.0, low=2_900.0, close=3_050.0,
                volume=500_000.0, currency="USDT",
            ),
        },
        macro=MacroData(vix=18.5, fear_greed_index=65.0),
    )


@pytest.fixture
def portfolio() -> PortfolioState:
    return PortfolioState(
        portfolio_id="port-multi-test",
        model=ModelProvider.GPT,
        architecture=AgentArchitecture.MULTI,
        market=Market.CRYPTO,
        cash=100_000.0,
        positions={},
        initial_capital=100_000.0,
    )


class TestMultiAgentPromptRouting:
    """Verify each stage sends its OWN system prompt to the LLM."""

    def test_analysts_receive_specialized_prompts(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> None:
        """4 analysts must each receive a different system prompt."""
        adapter = PromptCapturingAdapter()
        pipeline = MultiAgentPipeline(
            llm_adapter=adapter,
            model_provider=ModelProvider.GPT,
            debate_rounds=1,
            max_veto_retries=0,
        )

        loop = asyncio.new_event_loop()
        signal = loop.run_until_complete(pipeline.run(snapshot, portfolio))
        loop.close()

        # The first 4 prompts are the analysts
        analyst_prompts = adapter.captured_system_prompts[:4]
        assert len(analyst_prompts) == 4

        # Each analyst has a different prompt
        assert len(set(analyst_prompts)) == 4, (
            f"Expected 4 unique analyst prompts, got {len(set(analyst_prompts))}: "
            f"{[p[:40] for p in analyst_prompts]}"
        )

        # Verify each analyst type is present
        combined = " ".join(analyst_prompts).lower()
        assert "fundamental" in combined
        assert "technical" in combined
        assert "sentiment" in combined
        assert "macro" in combined or "news" in combined

    def test_debate_uses_bull_bear_prompts(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> None:
        """Bull and Bear researchers must receive distinct prompts."""
        adapter = PromptCapturingAdapter()
        pipeline = MultiAgentPipeline(
            llm_adapter=adapter,
            model_provider=ModelProvider.GPT,
            debate_rounds=1,
            max_veto_retries=0,
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(pipeline.run(snapshot, portfolio))
        loop.close()

        # After 4 analysts, prompts 4 and 5 are bull/bear
        debate_prompts = adapter.captured_system_prompts[4:6]
        assert len(debate_prompts) == 2

        combined = " ".join(debate_prompts).lower()
        assert "bull" in combined
        assert "bear" in combined

    def test_trader_risk_fund_manager_prompts(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> None:
        """Trader, Risk Manager, Fund Manager each get role-specific prompts."""
        adapter = PromptCapturingAdapter()
        pipeline = MultiAgentPipeline(
            llm_adapter=adapter,
            model_provider=ModelProvider.GPT,
            debate_rounds=1,
            max_veto_retries=0,
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(pipeline.run(snapshot, portfolio))
        loop.close()

        # After analysts(4) + debate(2) = indices 6,7,8 are trader, risk, fund manager
        later_prompts = adapter.captured_system_prompts[6:]
        combined = " ".join(later_prompts).lower()

        assert "trader" in combined
        assert "risk" in combined
        assert "fund manager" in combined or "final approval" in combined

    def test_pipeline_never_calls_generate_signal(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> None:
        """The fixed pipeline should use call_with_prompt(), not generate_signal()."""
        adapter = PromptCapturingAdapter()
        pipeline = MultiAgentPipeline(
            llm_adapter=adapter,
            model_provider=ModelProvider.GPT,
            debate_rounds=1,
            max_veto_retries=0,
        )

        loop = asyncio.new_event_loop()
        signal = loop.run_until_complete(pipeline.run(snapshot, portfolio))
        loop.close()

        # All calls should go through call_with_prompt
        assert len(adapter.captured_system_prompts) >= 8  # 4 analysts + 2 debate + trader + risk + fund
        # generate_signal should NOT have "BUG!" in any reasoning
        assert "BUG" not in signal.reasoning

    def test_pipeline_returns_valid_signal(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> None:
        """Pipeline output is a valid TradingSignal with MULTI architecture."""
        adapter = PromptCapturingAdapter()
        pipeline = MultiAgentPipeline(
            llm_adapter=adapter,
            model_provider=ModelProvider.GPT,
            debate_rounds=1,
            max_veto_retries=0,
        )

        loop = asyncio.new_event_loop()
        signal = loop.run_until_complete(pipeline.run(snapshot, portfolio))
        loop.close()

        assert signal.architecture == AgentArchitecture.MULTI
        assert signal.model == ModelProvider.GPT
        assert signal.market == Market.CRYPTO
        assert signal.reasoning  # Not empty
        assert "Debate Log" in signal.reasoning  # Audit trail appended

    def test_user_prompt_includes_context(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> None:
        """User prompts should contain market data and portfolio context."""
        adapter = PromptCapturingAdapter()
        pipeline = MultiAgentPipeline(
            llm_adapter=adapter,
            model_provider=ModelProvider.GPT,
            debate_rounds=1,
            max_veto_retries=0,
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(pipeline.run(snapshot, portfolio))
        loop.close()

        # Analyst user prompts should contain market data
        analyst_user = adapter.captured_user_prompts[0]
        assert "BTCUSDT" in analyst_user or "Market Data" in analyst_user

        # Trader user prompt should contain debate context
        trader_user = adapter.captured_user_prompts[6]
        assert "Debate" in trader_user or "Analyst" in trader_user
