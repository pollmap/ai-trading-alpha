"""Gemini 2.5 Pro LLM adapter using google-genai SDK."""

from __future__ import annotations

from google import genai
from google.genai import types as genai_types

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


class GeminiAdapter(BaseLLMAdapterImpl):
    """Gemini 2.5 Pro adapter.

    Uses google-genai SDK with JSON mode for structured output.
    1M token context allows bulk processing.
    Grounding disabled for benchmark fairness.
    """

    def __init__(
        self,
        architecture: AgentArchitecture,
        cost_tracker: CostTracker,
        call_logger: LLMCallLogger,
        model_name: str = "gemini-2.5-pro-preview-06-05",
        timeout_seconds: int = 60,
    ) -> None:
        super().__init__(
            model_provider=ModelProvider.GEMINI,
            architecture=architecture,
            cost_tracker=cost_tracker,
            call_logger=call_logger,
            timeout_seconds=timeout_seconds,
        )
        settings = get_settings()
        self._client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value(),
        )
        self._model_name = model_name

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int, int]:
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        content = response.text or ""

        # Extract token usage
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        cached_tokens = usage.cached_content_token_count if usage else 0

        return content, input_tokens, output_tokens, cached_tokens

    def _build_system_prompt(self, market: Market) -> str:
        return build_system_prompt(market, output_format="json")

    def _build_user_prompt(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState,
    ) -> str:
        return build_user_prompt(snapshot, portfolio)
