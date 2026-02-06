"""Claude Sonnet 4.5 LLM adapter using Anthropic SDK."""

from __future__ import annotations

from anthropic import AsyncAnthropic

from config.settings import get_settings
from src.core.logging import get_logger
from src.core.types import (
    AgentArchitecture,
    Market,
    MarketSnapshot,
    ModelProvider,
    PortfolioState,
)
from src.llm.base import BaseLLMAdapterImpl
from src.llm.call_logger import LLMCallLogger
from src.llm.cost_tracker import CostTracker
from src.llm.prompt_templates.single_agent import build_system_prompt, build_user_prompt

log = get_logger(__name__)


class ClaudeAdapter(BaseLLMAdapterImpl):
    """Claude Sonnet 4.5 adapter.

    Uses Anthropic SDK with:
    - Prompt caching (90% cost reduction on system prompt)
    - XML tag-based structured output
    - Optional extended thinking for complex analysis
    """

    def __init__(
        self,
        architecture: AgentArchitecture,
        cost_tracker: CostTracker,
        call_logger: LLMCallLogger,
        model_name: str = "claude-sonnet-4-5-20250514",
        timeout_seconds: int = 60,
        extended_thinking: bool = False,
    ) -> None:
        super().__init__(
            model_provider=ModelProvider.CLAUDE,
            architecture=architecture,
            cost_tracker=cost_tracker,
            call_logger=call_logger,
            timeout_seconds=timeout_seconds,
        )
        settings = get_settings()
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
        )
        self._model_name = model_name
        self._extended_thinking = extended_thinking

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int, int]:
        # System prompt with cache control for cost reduction
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        kwargs: dict = {
            "model": self._model_name,
            "max_tokens": 4096,
            "system": system_blocks,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        if not self._extended_thinking:
            kwargs["temperature"] = 0.0

        response = await self._client.messages.create(**kwargs)

        # Extract text content
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        # Token usage
        usage = response.usage
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        cached_tokens = getattr(usage, "cache_read_input_tokens", 0)

        return content, input_tokens, output_tokens, cached_tokens

    def _build_system_prompt(self, market: Market) -> str:
        return build_system_prompt(market, output_format="xml")

    def _build_user_prompt(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> str:
        return build_user_prompt(snapshot, portfolio)
