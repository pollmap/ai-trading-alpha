"""GPT-4o-mini LLM adapter (reference baseline) using OpenAI SDK."""

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


class GPTAdapter(BaseLLMAdapterImpl):
    """GPT-4o-mini adapter — reference baseline for comparison with existing research.

    Structurally identical to DeepSeek adapter (both use OpenAI SDK).
    """

    def __init__(
        self,
        architecture: AgentArchitecture,
        cost_tracker: CostTracker,
        call_logger: LLMCallLogger,
        model_name: str = "gpt-4o-mini",
        timeout_seconds: int = 60,
    ) -> None:
        super().__init__(
            model_provider=ModelProvider.GPT,
            architecture=architecture,
            cost_tracker=cost_tracker,
            call_logger=call_logger,
            timeout_seconds=timeout_seconds,
        )
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
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
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return content, input_tokens, output_tokens, 0

    def _build_system_prompt(self, market: Market) -> str:
        return build_system_prompt(market, output_format="json")

    def _build_user_prompt(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> str:
        return build_user_prompt(snapshot, portfolio)
