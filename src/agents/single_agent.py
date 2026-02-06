"""Single agent — simplest form: 1 LLM call -> 1 TradingSignal."""

from __future__ import annotations

from src.core.interfaces import BaseLLMAdapter
from src.core.logging import get_logger
from src.core.types import MarketSnapshot, PortfolioState, TradingSignal

log = get_logger(__name__)


class SingleAgent:
    """Single agent that generates one trading signal per cycle.

    This is the baseline agent architecture:
    1. Receives MarketSnapshot + PortfolioState
    2. Makes 1 LLM call
    3. Returns 1 TradingSignal

    Purpose: Baseline for comparison against multi-agent pipeline.
    """

    def __init__(self, llm_adapter: BaseLLMAdapter) -> None:
        self._adapter = llm_adapter

    async def run(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TradingSignal:
        """Execute a single trading cycle.

        Args:
            snapshot: Current market data.
            portfolio: Current portfolio state.

        Returns:
            TradingSignal with action, weight, confidence, and reasoning.
        """
        log.info(
            "single_agent_running",
            model=portfolio.model.value,
            market=snapshot.market.value,
            symbols=len(snapshot.symbols),
        )

        signal = await self._adapter.generate_signal(snapshot, portfolio)

        log.info(
            "single_agent_complete",
            model=portfolio.model.value,
            action=signal.action.value,
            symbol=signal.symbol,
            confidence=f"{signal.confidence:.2f}",
            latency_ms=f"{signal.latency_ms:.0f}",
        )

        return signal
