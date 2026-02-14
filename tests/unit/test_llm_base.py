"""Tests for BaseLLMAdapterImpl — retry, timeout, fallback, call_with_prompt."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from uuid_extensions import uuid7

from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    MacroData,
    MarketSnapshot,
    ModelProvider,
    PortfolioState,
    SymbolData,
    TradingSignal,
)
from src.llm.base import BaseLLMAdapterImpl
from src.llm.call_logger import LLMCallLogger
from src.llm.cost_tracker import CostTracker


# -- Test Adapter (concrete subclass for testing) --


class _TestAdapter(BaseLLMAdapterImpl):
    """Concrete adapter that records calls and can be configured to fail."""

    def __init__(
        self,
        responses: list[str | Exception] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_provider=ModelProvider.GPT,
            architecture=AgentArchitecture.SINGLE,
            cost_tracker=kwargs.get("cost_tracker") or CostTracker(),
            call_logger=kwargs.get("call_logger") or LLMCallLogger(),
            timeout_seconds=kwargs.get("timeout_seconds", 5),
            max_retries=kwargs.get("max_retries", 3),
        )
        self._responses = list(responses or ['{"action":"BUY","symbol":"AAPL","weight":0.25,"confidence":0.8,"reasoning":"test"}'])
        self._call_count = 0
        self._received_prompts: list[tuple[str, str]] = []

    async def _call_llm(
        self, system_prompt: str, user_prompt: str,
    ) -> tuple[str, int, int, int]:
        self._received_prompts.append((system_prompt, user_prompt))
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        response = self._responses[idx]
        if isinstance(response, Exception):
            raise response
        return response, 100, 50, 0

    def _build_system_prompt(self, market: Market) -> str:
        return f"Default system prompt for {market.value}"

    def _build_user_prompt(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> str:
        return "Default user prompt"


# -- Fixtures --


@pytest.fixture
def snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id="snap-test",
        timestamp=datetime.now(timezone.utc),
        market=Market.US,
        symbols={
            "AAPL": SymbolData(
                symbol="AAPL", market=Market.US,
                open=185.0, high=187.0, low=184.0, close=186.0,
                volume=50_000_000.0, currency="USD",
            ),
        },
        macro=MacroData(),
    )


@pytest.fixture
def portfolio() -> PortfolioState:
    return PortfolioState(
        portfolio_id="port-test",
        model=ModelProvider.GPT,
        architecture=AgentArchitecture.SINGLE,
        market=Market.US,
        cash=100_000.0,
        positions={},
        initial_capital=100_000.0,
    )


# -- Tests --


class TestGenerateSignal:
    """Test generate_signal() — the standard single-agent flow."""

    def test_successful_call(self, snapshot, portfolio) -> None:
        """Normal successful call returns a parsed signal."""
        adapter = _TestAdapter()
        loop = asyncio.new_event_loop()
        signal = loop.run_until_complete(adapter.generate_signal(snapshot, portfolio))
        loop.close()

        assert signal.action == Action.BUY
        assert signal.symbol == "AAPL"
        assert adapter._call_count == 1

    def test_uses_default_prompts(self, snapshot, portfolio) -> None:
        """generate_signal() uses _build_system_prompt / _build_user_prompt."""
        adapter = _TestAdapter()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(adapter.generate_signal(snapshot, portfolio))
        loop.close()

        sys_prompt, user_prompt = adapter._received_prompts[0]
        assert "US" in sys_prompt  # Default system prompt includes market
        assert user_prompt == "Default user prompt"

    def test_retry_on_failure(self, snapshot, portfolio) -> None:
        """Adapter retries on exception and succeeds on second try."""
        adapter = _TestAdapter(
            responses=[
                RuntimeError("API error"),
                '{"action":"HOLD","symbol":"AAPL","weight":0.0,"confidence":0.5,"reasoning":"retry success"}',
            ],
            max_retries=3,
            timeout_seconds=1,
        )
        loop = asyncio.new_event_loop()
        signal = loop.run_until_complete(adapter.generate_signal(snapshot, portfolio))
        loop.close()

        assert adapter._call_count == 2
        assert signal.action in (Action.BUY, Action.SELL, Action.HOLD)

    def test_all_retries_exhausted_returns_hold(self, snapshot, portfolio) -> None:
        """After max retries, adapter returns HOLD fallback."""
        adapter = _TestAdapter(
            responses=[RuntimeError("always fail")],
            max_retries=2,
            timeout_seconds=1,
        )
        loop = asyncio.new_event_loop()
        signal = loop.run_until_complete(adapter.generate_signal(snapshot, portfolio))
        loop.close()

        assert signal.action == Action.HOLD
        assert "failed" in signal.reasoning.lower()
        assert adapter._call_count == 2

    def test_cost_tracking(self, snapshot, portfolio) -> None:
        """Cost tracker records the call."""
        cost_tracker = CostTracker()
        adapter = _TestAdapter(cost_tracker=cost_tracker)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(adapter.generate_signal(snapshot, portfolio))
        loop.close()

        records = cost_tracker.get_records()
        assert len(records) == 1
        assert records[0].model == "gpt"
        assert records[0].input_tokens == 100
        assert records[0].output_tokens == 50

    def test_call_logging(self, snapshot, portfolio) -> None:
        """Call logger records the prompt and response."""
        call_logger = LLMCallLogger()
        adapter = _TestAdapter(call_logger=call_logger)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(adapter.generate_signal(snapshot, portfolio))
        loop.close()

        records = call_logger.get_records()
        assert len(records) == 1
        assert "US" in records[0].prompt_text
        assert records[0].parsed_success is True


class TestCallWithPrompt:
    """Test call_with_prompt() — custom system/user prompts."""

    def test_custom_prompts_passed_to_llm(self, snapshot, portfolio) -> None:
        """call_with_prompt() passes custom prompts directly to _call_llm."""
        adapter = _TestAdapter()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            adapter.call_with_prompt(
                system_prompt="You are a Fundamental Analyst",
                user_prompt="Analyze AAPL fundamentals",
                snapshot=snapshot,
                portfolio=portfolio,
            ),
        )
        loop.close()

        sys_prompt, user_prompt = adapter._received_prompts[0]
        assert sys_prompt == "You are a Fundamental Analyst"
        assert user_prompt == "Analyze AAPL fundamentals"

    def test_call_with_prompt_retries(self, snapshot, portfolio) -> None:
        """call_with_prompt() also retries on failure."""
        adapter = _TestAdapter(
            responses=[
                RuntimeError("fail once"),
                '{"action":"SELL","symbol":"AAPL","weight":0.1,"confidence":0.6,"reasoning":"recovered"}',
            ],
            max_retries=3,
            timeout_seconds=1,
        )
        loop = asyncio.new_event_loop()
        signal = loop.run_until_complete(
            adapter.call_with_prompt(
                system_prompt="Custom role",
                user_prompt="Custom data",
                snapshot=snapshot,
                portfolio=portfolio,
            ),
        )
        loop.close()

        assert adapter._call_count == 2
        assert signal.action in (Action.BUY, Action.SELL, Action.HOLD)

    def test_call_with_prompt_fallback_hold(self, snapshot, portfolio) -> None:
        """call_with_prompt() returns HOLD after all retries exhausted."""
        adapter = _TestAdapter(
            responses=[RuntimeError("always fail")],
            max_retries=2,
            timeout_seconds=1,
        )
        loop = asyncio.new_event_loop()
        signal = loop.run_until_complete(
            adapter.call_with_prompt(
                system_prompt="Any role",
                user_prompt="Any data",
                snapshot=snapshot,
                portfolio=portfolio,
            ),
        )
        loop.close()

        assert signal.action == Action.HOLD
        assert adapter._call_count == 2

    def test_different_prompts_for_different_roles(self, snapshot, portfolio) -> None:
        """Each multi-agent role should receive its own unique prompt."""
        adapter = _TestAdapter()
        loop = asyncio.new_event_loop()

        # Simulate calling with different analyst prompts
        loop.run_until_complete(
            adapter.call_with_prompt(
                system_prompt="You are a Fundamental Analyst",
                user_prompt="Analyze AAPL",
                snapshot=snapshot,
                portfolio=portfolio,
            ),
        )
        loop.run_until_complete(
            adapter.call_with_prompt(
                system_prompt="You are a Technical Analyst",
                user_prompt="Analyze AAPL",
                snapshot=snapshot,
                portfolio=portfolio,
            ),
        )
        loop.close()

        # Verify different system prompts were sent
        assert adapter._received_prompts[0][0] == "You are a Fundamental Analyst"
        assert adapter._received_prompts[1][0] == "You are a Technical Analyst"
        assert adapter._received_prompts[0][0] != adapter._received_prompts[1][0]


class TestFallbackHold:
    """Test the HOLD fallback signal."""

    def test_fallback_has_required_fields(self, snapshot) -> None:
        """Fallback HOLD signal has all required TradingSignal fields."""
        adapter = _TestAdapter()
        signal = adapter._fallback_hold(snapshot, "AAPL")

        assert signal.action == Action.HOLD
        assert signal.weight == 0.0
        assert signal.confidence == 0.0
        assert signal.symbol == "AAPL"
        assert signal.market == Market.US
        assert signal.model == ModelProvider.GPT
        assert signal.reasoning  # Must not be empty
