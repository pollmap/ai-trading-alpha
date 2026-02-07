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
