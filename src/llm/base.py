"""Base LLM adapter with common logic (retry, timeout, cost tracking, call logging)."""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from datetime import datetime, timezone

from uuid_extensions import uuid7

from src.core.exceptions import LLMResponseError, LLMTimeoutError
from src.core.interfaces import BaseLLMAdapter
from src.core.logging import get_logger
from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    MarketSnapshot,
    ModelProvider,
    PortfolioState,
    TradingSignal,
)
from src.llm.call_logger import LLMCallLogger
from src.llm.cost_tracker import CostTracker
from src.llm.response_parser import ResponseParser

log = get_logger(__name__)


class BaseLLMAdapterImpl(BaseLLMAdapter):
    """Base implementation with retry, timeout, cost tracking, and call logging.

    Subclasses implement _call_llm() for provider-specific API calls.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        architecture: AgentArchitecture,
        cost_tracker: CostTracker,
        call_logger: LLMCallLogger,
        timeout_seconds: int = 60,
        max_retries: int = 3,
    ) -> None:
        self._model_provider = model_provider
        self._architecture = architecture
        self._cost_tracker = cost_tracker
        self._call_logger = call_logger
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._parser = ResponseParser()

    @abstractmethod
    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int, int]:
        """Provider-specific LLM API call.

        Returns:
            Tuple of (raw_response_text, input_tokens, output_tokens, cached_tokens)
        """
        ...

    @abstractmethod
    def _build_system_prompt(self, market: Market) -> str:
        """Build the system prompt for this adapter."""
        ...

    @abstractmethod
    def _build_user_prompt(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> str:
        """Build the user prompt from market data and portfolio."""
        ...

    async def generate_signal(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TradingSignal:
        """Generate a trading signal with retry, timeout, cost tracking, and logging."""
        system_prompt = self._build_system_prompt(snapshot.market)
        user_prompt = self._build_user_prompt(snapshot, portfolio)
        full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

        last_error: Exception | None = None
        symbol = next(iter(snapshot.symbols), "UNKNOWN")

        for attempt in range(1, self._max_retries + 1):
            try:
                signal = await self._attempt_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    full_prompt=full_prompt,
                    snapshot=snapshot,
                    portfolio=portfolio,
                    symbol=symbol,
                    attempt=attempt,
                )
                return signal

            except asyncio.TimeoutError:
                last_error = LLMTimeoutError(
                    f"Timeout after {self._timeout}s",
                    context={"model": self._model_provider.value, "attempt": attempt},
                )
                log.warning(
                    "llm_timeout",
                    model=self._model_provider.value,
                    attempt=attempt,
                    timeout=self._timeout,
                )

            except Exception as exc:
                last_error = exc
                log.warning(
                    "llm_call_failed",
                    model=self._model_provider.value,
                    attempt=attempt,
                    error=str(exc),
                )

            # Exponential backoff between retries
            if attempt < self._max_retries:
                wait = 2 ** attempt
                await asyncio.sleep(wait)

        # All retries exhausted — return HOLD
        log.error(
            "llm_all_retries_exhausted",
            model=self._model_provider.value,
            error=str(last_error),
        )
        return self._fallback_hold(snapshot, symbol)

    async def _attempt_call(
        self,
        system_prompt: str,
        user_prompt: str,
        full_prompt: str,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
        symbol: str,
        attempt: int,
    ) -> TradingSignal:
        """Single attempt: call LLM, parse, track cost, log call."""
        async with self._cost_tracker.track(self._model_provider.value) as cost_acc:
            async with self._call_logger.log_call(
                self._model_provider.value,
                self._architecture.value,
            ) as call_acc:
                call_acc.set_prompt(full_prompt)

                # Call with timeout
                raw_response, in_tok, out_tok, cache_tok = await asyncio.wait_for(
                    self._call_llm(system_prompt, user_prompt),
                    timeout=self._timeout,
                )

                cost_acc.set_usage(in_tok, out_tok, cache_tok)
                call_acc.set_usage(in_tok, out_tok)

                # Parse response
                signal, parsed_ok = self._parser.parse_signal(
                    raw_response=raw_response,
                    model=self._model_provider,
                    architecture=self._architecture,
                    snapshot_id=snapshot.snapshot_id,
                    symbol=symbol,
                    market=snapshot.market,
                )
                call_acc.set_response(raw_response, parsed_ok)

                signal.latency_ms = cost_acc.elapsed_ms()
                signal.token_usage = {
                    "input": in_tok,
                    "output": out_tok,
                    "cached": cache_tok,
                }

                return signal

    def _fallback_hold(
        self, snapshot: MarketSnapshot, symbol: str,
    ) -> TradingSignal:
        """Return HOLD signal after all retries exhausted."""
        return TradingSignal(
            signal_id=str(uuid7()),
            snapshot_id=snapshot.snapshot_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            market=snapshot.market,
            action=Action.HOLD,
            weight=0.0,
            confidence=0.0,
            reasoning="All LLM call attempts failed — defaulting to HOLD",
            model=self._model_provider,
            architecture=self._architecture,
        )
