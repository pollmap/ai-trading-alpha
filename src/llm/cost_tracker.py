"""LLM API cost tracking middleware."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator

from uuid7 import uuid7

from src.core.logging import get_logger

log = get_logger(__name__)

# ── Pricing Table (USD per million tokens) ───────────────────────

PRICING: dict[str, dict[str, float]] = {
    "deepseek": {
        "input": 0.55,
        "output": 2.19,
        "cache_hit": 0.07,
    },
    "gemini": {
        "input": 1.25,
        "output": 10.0,
        "cache_hit": 0.0,
    },
    "claude": {
        "input": 3.0,
        "output": 15.0,
        "cache_hit": 0.30,
    },
    "gpt": {
        "input": 0.15,
        "output": 0.60,
        "cache_hit": 0.0,
    },
}


@dataclass
class CostRecord:
    """Single LLM call cost record."""

    log_id: str
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass
class CostAccumulator:
    """Mutable accumulator used during a tracked call."""

    model: str
    start_time: float = field(default_factory=time.monotonic)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def set_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens

    def calculate_cost(self) -> float:
        pricing = PRICING.get(self.model, PRICING["gpt"])
        cost = (
            (self.input_tokens - self.cached_tokens) * pricing["input"] / 1_000_000
            + self.cached_tokens * pricing["cache_hit"] / 1_000_000
            + self.output_tokens * pricing["output"] / 1_000_000
        )
        return max(0.0, cost)

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000


class CostTracker:
    """Track all LLM API call costs.

    Usage:
        tracker = CostTracker()
        async with tracker.track("deepseek") as acc:
            response = await llm_call(...)
            acc.set_usage(input_tokens=100, output_tokens=50)
        # Cost is automatically logged and stored
    """

    def __init__(self) -> None:
        self._records: list[CostRecord] = []
        self._db_writer: object | None = None  # Set externally for DB persistence

    @asynccontextmanager
    async def track(self, model: str) -> AsyncIterator[CostAccumulator]:
        """Context manager to track a single LLM call."""
        acc = CostAccumulator(model=model)
        try:
            yield acc
        finally:
            cost = acc.calculate_cost()
            latency = acc.elapsed_ms()

            record = CostRecord(
                log_id=str(uuid7()),
                timestamp=datetime.now(timezone.utc),
                model=model,
                input_tokens=acc.input_tokens,
                output_tokens=acc.output_tokens,
                cached_tokens=acc.cached_tokens,
                cost_usd=cost,
                latency_ms=latency,
            )
            self._records.append(record)

            log.info(
                "llm_cost_tracked",
                model=model,
                input_tokens=acc.input_tokens,
                output_tokens=acc.output_tokens,
                cached_tokens=acc.cached_tokens,
                cost_usd=f"${cost:.6f}",
                latency_ms=f"{latency:.0f}",
            )

    # ── Aggregation ──────────────────────────────────────────────

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self._records)

    @property
    def total_calls(self) -> int:
        return len(self._records)

    def cost_by_model(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for r in self._records:
            result[r.model] = result.get(r.model, 0.0) + r.cost_usd
        return result

    def get_records(self) -> list[CostRecord]:
        return list(self._records)

    def summary(self) -> dict[str, object]:
        return {
            "total_calls": self.total_calls,
            "total_cost_usd": f"${self.total_cost_usd:.4f}",
            "by_model": {
                k: f"${v:.4f}" for k, v in self.cost_by_model().items()
            },
        }
