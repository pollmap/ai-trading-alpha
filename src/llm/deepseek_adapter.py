"""DeepSeek R1 LLM adapter using OpenAI-compatible SDK."""

from __future__ import annotations

from openai import AsyncOpenAI

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


class DeepSeekAdapter(BaseLLMAdapterImpl):
    """DeepSeek R1 adapter.

    Uses OpenAI SDK with custom base_url.
    Extracts reasoning_content for Chain-of-Thought audit trail.
    """

    def __init__(
        self,
        architecture: AgentArchitecture,
        cost_tracker: CostTracker,
        call_logger: LLMCallLogger,
        model_name: str = "deepseek-reasoner",
        timeout_seconds: int = 60,
    ) -> None:
        super().__init__(
            model_provider=ModelProvider.DEEPSEEK,
            architecture=architecture,
            cost_tracker=cost_tracker,
            call_logger=call_logger,
            timeout_seconds=timeout_seconds,
        )
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.deepseek_api_key.get_secret_value(),
            base_url="https://api.deepseek.com",
        )
        self._model_name = model_name

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int, int]:
        response = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        # Extract Chain-of-Thought if available
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        if reasoning_content:
            log.debug("deepseek_cot_extracted", cot_length=len(reasoning_content))

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cached_tokens = getattr(usage, "prompt_cache_hit_tokens", 0) if usage else 0

        return content, input_tokens, output_tokens, cached_tokens

    def _build_system_prompt(self, market: Market) -> str:
        return build_system_prompt(market, output_format="json")

    def _build_user_prompt(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> str:
        return build_user_prompt(snapshot, portfolio)
