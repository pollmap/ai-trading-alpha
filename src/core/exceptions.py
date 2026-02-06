"""Custom exception hierarchy for ATLAS."""

from __future__ import annotations

from typing import Any


class ATLASBaseError(Exception):
    """Base exception for all ATLAS errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}


# ── Data Layer ───────────────────────────────────────────────────

class DataFetchError(ATLASBaseError):
    """External API call failed after retries."""


class SnapshotStaleError(ATLASBaseError):
    """MarketSnapshot data is older than the forward-fill threshold."""


class RateLimitError(ATLASBaseError):
    """API rate limit exceeded."""


# ── LLM Layer ────────────────────────────────────────────────────

class LLMResponseError(ATLASBaseError):
    """LLM response parsing failed after max retries."""


class LLMTimeoutError(ATLASBaseError):
    """LLM call exceeded the configured timeout."""


# ── Simulator Layer ──────────────────────────────────────────────

class InsufficientFundsError(ATLASBaseError):
    """Not enough cash to execute the order."""


class PositionLimitError(ATLASBaseError):
    """Order would exceed max position weight or violate min cash ratio."""


# ── Agent Layer ──────────────────────────────────────────────────

class AgentExecutionError(ATLASBaseError):
    """Agent pipeline failed during execution."""


class OrchestratorError(ATLASBaseError):
    """Benchmark orchestrator encountered a critical failure."""
