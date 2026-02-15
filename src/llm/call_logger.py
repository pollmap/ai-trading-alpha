"""LLM call logger — records full prompt/response for reproducibility."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator

from uuid_extensions import uuid7

from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class CallRecord:
    """Complete record of a single LLM call."""

    call_id: str
    signal_id: str | None
    timestamp: datetime
    model: str
    role: str  # 'single', 'analyst_fundamental', 'researcher_bull', etc.
    prompt_text: str
    raw_response: str
    parsed_success: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int


@dataclass
class CallAccumulator:
    """Mutable accumulator used during a logged call."""

    model: str
    role: str
    signal_id: str | None = None
    prompt_text: str = ""
    raw_response: str = ""
    parsed_success: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    start_time: float = field(default_factory=time.monotonic)

    def set_prompt(self, text: str) -> None:
        self.prompt_text = text

    def set_response(self, text: str, parsed_ok: bool = True) -> None:
        self.raw_response = text
        self.parsed_success = parsed_ok

    def set_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000


class LLMCallLogger:
    """Log full prompt/response for every LLM call.

    Usage:
        logger = LLMCallLogger()
        async with logger.log_call("deepseek", "single") as acc:
            acc.set_prompt(prompt_text)
            response = await llm_call(prompt_text)
            acc.set_response(response, parsed_ok=True)
            acc.set_usage(100, 50)
    """

    _MAX_RECORDS = 10_000  # Prevent unbounded memory growth

    def __init__(self) -> None:
        self._records: list[CallRecord] = []
        self._total_calls: int = 0

    @asynccontextmanager
    async def log_call(
        self,
        model: str,
        role: str,
        signal_id: str | None = None,
    ) -> AsyncIterator[CallAccumulator]:
        """Context manager to log a complete LLM call."""
        acc = CallAccumulator(model=model, role=role, signal_id=signal_id)
        try:
            yield acc
        finally:
            record = CallRecord(
                call_id=str(uuid7()),
                signal_id=acc.signal_id,
                timestamp=datetime.now(timezone.utc),
                model=model,
                role=role,
                prompt_text=acc.prompt_text,
                raw_response=acc.raw_response[:10000],  # Truncate extremely long responses
                parsed_success=acc.parsed_success,
                latency_ms=acc.elapsed_ms(),
                input_tokens=acc.input_tokens,
                output_tokens=acc.output_tokens,
            )
            self._records.append(record)
            self._total_calls += 1
            if len(self._records) > self._MAX_RECORDS:
                self._records = self._records[-self._MAX_RECORDS:]

            log.info(
                "llm_call_logged",
                model=model,
                role=role,
                parsed_success=acc.parsed_success,
                latency_ms=f"{acc.elapsed_ms():.0f}",
                prompt_len=len(acc.prompt_text),
                response_len=len(acc.raw_response),
            )

    def get_records(self) -> list[CallRecord]:
        return list(self._records)

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def parse_success_rate(self) -> float:
        if not self._records:
            return 1.0
        successes = sum(1 for r in self._records if r.parsed_success)
        return successes / len(self._records)
