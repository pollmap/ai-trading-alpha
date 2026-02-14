"""Abstract base classes — all modules must implement these interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.core.types import (
    MarketSnapshot,
    PortfolioState,
    SymbolData,
    TradingSignal,
)


class BaseLLMAdapter(ABC):
    """Interface for all LLM provider adapters."""

    @abstractmethod
    async def generate_signal(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TradingSignal:
        """Generate a trading signal from market data and portfolio state."""
        ...

    async def call_with_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TradingSignal:
        """Generate a signal using custom system/user prompts.

        Used by the multi-agent pipeline so each role (analyst, trader,
        risk manager, fund manager) can inject its own prompt while
        reusing the adapter's retry, timeout, and cost-tracking logic.

        Default implementation falls back to ``generate_signal()``.
        """
        return await self.generate_signal(snapshot, portfolio)


class BaseMarketDataAdapter(ABC):
    """Interface for all market data source adapters."""

    @abstractmethod
    async def fetch_latest(self) -> dict[str, SymbolData]:
        """Fetch the latest market data for configured symbols."""
        ...


class BaseMetricsCalculator(ABC):
    """Interface for performance metrics calculators."""

    @abstractmethod
    async def calculate(
        self, portfolio_history: list[PortfolioState],
    ) -> dict[str, Any]:
        """Calculate performance metrics from portfolio history."""
        ...
