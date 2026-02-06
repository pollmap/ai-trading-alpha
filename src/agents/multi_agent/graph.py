"""LangGraph multi-agent workflow definition."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, TypedDict

from uuid_extensions import uuid7

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

log = get_logger(__name__)


# ── State Definition (TypedDict for LangGraph compatibility) ─────

class MultiAgentState(TypedDict, total=False):
    """State flowing through the multi-agent pipeline."""
    snapshot: MarketSnapshot
    portfolio: PortfolioState
    model: str
    architecture: str
    # Stage 1: Analyst outputs
    fundamental_report: str
    technical_report: str
    sentiment_report: str
    news_report: str
    # Stage 2: Debate
    bull_case: str
    bear_case: str
    debate_log: list[str]
    # Stage 3: Trade proposal
    trade_proposal: str
    # Stage 4: Risk assessment
    risk_assessment: str
    risk_approved: bool
    risk_veto_count: int
    # Stage 5: Final signal
    final_signal: dict[str, Any] | None


class MultiAgentPipeline:
    """Multi-agent trading pipeline.

    Flow:
    1. 4 Analysts run in parallel (fundamental, technical, sentiment, news)
    2. Bull/Bear researchers debate (2 rounds)
    3. Trader makes decision
    4. Risk Manager evaluates (VETO -> back to Trader, max 2 retries)
    5. Fund Manager gives final approval

    Each node uses the same LLM adapter (same model provider).
    """

    def __init__(
        self,
        llm_adapter: BaseLLMAdapter,
        model_provider: ModelProvider,
        debate_rounds: int = 2,
        max_veto_retries: int = 2,
    ) -> None:
        self._adapter = llm_adapter
        self._model_provider = model_provider
        self._debate_rounds = debate_rounds
        self._max_veto_retries = max_veto_retries

    async def run(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> TradingSignal:
        """Execute the full multi-agent pipeline."""
        log.info(
            "multi_agent_pipeline_start",
            model=self._model_provider.value,
            market=snapshot.market.value,
        )

        state: MultiAgentState = {
            "snapshot": snapshot,
            "portfolio": portfolio,
            "model": self._model_provider.value,
            "architecture": "multi",
            "debate_log": [],
            "risk_veto_count": 0,
            "risk_approved": False,
            "final_signal": None,
        }

        # Stage 1: Parallel analysts
        state = await self._run_analysts(state)

        # Stage 2: Bull/Bear debate
        state = await self._run_debate(state)

        # Stage 3-4: Trader + Risk loop
        for attempt in range(self._max_veto_retries + 1):
            state = await self._run_trader(state)
            state = await self._run_risk_manager(state)

            if state.get("risk_approved", False):
                break

            state["risk_veto_count"] = state.get("risk_veto_count", 0) + 1
            log.info(
                "risk_veto",
                model=self._model_provider.value,
                attempt=attempt + 1,
            )

        # Stage 5: Fund Manager or forced HOLD
        if state.get("risk_approved", False):
            state = await self._run_fund_manager(state)
        else:
            log.warning("max_veto_reached_forcing_hold", model=self._model_provider.value)

        return self._build_signal(state, snapshot)

    async def _run_analysts(self, state: MultiAgentState) -> MultiAgentState:
        """Run 4 analysts in parallel."""
        from src.llm.prompt_templates.analyst import (
            build_fundamental_prompt,
            build_news_prompt,
            build_sentiment_prompt,
            build_technical_prompt,
        )

        snapshot = state["snapshot"]
        market = snapshot.market
        market_data = snapshot.to_prompt_summary()
        portfolio_data = state["portfolio"].to_prompt_summary()

        async def call_analyst(system_prompt: str) -> str:
            user_prompt = f"Market Data:\n{market_data}\n\nPortfolio:\n{portfolio_data}"
            signal = await self._adapter.generate_signal(snapshot, state["portfolio"])
            return signal.reasoning

        tasks = [
            call_analyst(build_fundamental_prompt(market)),
            call_analyst(build_technical_prompt(market)),
            call_analyst(build_sentiment_prompt(market)),
            call_analyst(build_news_prompt(market)),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        state["fundamental_report"] = str(results[0]) if not isinstance(results[0], Exception) else "Analysis failed"
        state["technical_report"] = str(results[1]) if not isinstance(results[1], Exception) else "Analysis failed"
        state["sentiment_report"] = str(results[2]) if not isinstance(results[2], Exception) else "Analysis failed"
        state["news_report"] = str(results[3]) if not isinstance(results[3], Exception) else "Analysis failed"

        log.info("analysts_complete", model=self._model_provider.value)
        return state

    async def _run_debate(self, state: MultiAgentState) -> MultiAgentState:
        """Run Bull/Bear debate for configured rounds."""
        reports = (
            f"Fundamental: {state.get('fundamental_report', 'N/A')}\n"
            f"Technical: {state.get('technical_report', 'N/A')}\n"
            f"Sentiment: {state.get('sentiment_report', 'N/A')}\n"
            f"News: {state.get('news_report', 'N/A')}"
        )

        # Initial cases
        bull_signal = await self._adapter.generate_signal(state["snapshot"], state["portfolio"])
        state["bull_case"] = f"BULL: {bull_signal.reasoning}"

        bear_signal = await self._adapter.generate_signal(state["snapshot"], state["portfolio"])
        state["bear_case"] = f"BEAR: {bear_signal.reasoning}"

        debate_log = state.get("debate_log", [])
        debate_log.append(f"Round 1 Bull: {state['bull_case'][:500]}")
        debate_log.append(f"Round 1 Bear: {state['bear_case'][:500]}")

        # Additional debate rounds
        for round_num in range(2, self._debate_rounds + 1):
            rebuttal = await self._adapter.generate_signal(state["snapshot"], state["portfolio"])
            debate_log.append(f"Round {round_num}: {rebuttal.reasoning[:500]}")

        state["debate_log"] = debate_log
        log.info("debate_complete", model=self._model_provider.value, rounds=self._debate_rounds)
        return state

    async def _run_trader(self, state: MultiAgentState) -> MultiAgentState:
        """Trader synthesizes debate into trade proposal."""
        signal = await self._adapter.generate_signal(state["snapshot"], state["portfolio"])
        state["trade_proposal"] = signal.reasoning
        log.info("trader_decision_complete", model=self._model_provider.value)
        return state

    async def _run_risk_manager(self, state: MultiAgentState) -> MultiAgentState:
        """Risk Manager evaluates the trade proposal."""
        signal = await self._adapter.generate_signal(state["snapshot"], state["portfolio"])

        # Parse risk decision from reasoning
        approved = "approve" in signal.reasoning.lower() or signal.action != Action.HOLD
        state["risk_assessment"] = signal.reasoning
        state["risk_approved"] = approved

        log.info(
            "risk_assessment_complete",
            model=self._model_provider.value,
            approved=approved,
        )
        return state

    async def _run_fund_manager(self, state: MultiAgentState) -> MultiAgentState:
        """Fund Manager gives final approval."""
        signal = await self._adapter.generate_signal(state["snapshot"], state["portfolio"])

        state["final_signal"] = {
            "action": signal.action.value,
            "symbol": signal.symbol,
            "weight": signal.weight,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
        }

        log.info("fund_manager_complete", model=self._model_provider.value)
        return state

    def _build_signal(
        self, state: MultiAgentState, snapshot: MarketSnapshot,
    ) -> TradingSignal:
        """Build final TradingSignal from pipeline state."""
        final = state.get("final_signal")

        if final:
            action = Action(final.get("action", "HOLD"))
            symbol = final.get("symbol", next(iter(snapshot.symbols), "UNKNOWN"))
            weight = float(final.get("weight", 0.0))
            confidence = float(final.get("confidence", 0.0))
            reasoning = str(final.get("reasoning", ""))
        else:
            action = Action.HOLD
            symbol = next(iter(snapshot.symbols), "UNKNOWN")
            weight = 0.0
            confidence = 0.0
            reasoning = "Pipeline did not produce a final signal — defaulting to HOLD"

        # Append debate log to reasoning for audit trail
        debate = state.get("debate_log", [])
        if debate:
            reasoning += "\n\n--- Debate Log ---\n" + "\n".join(debate[:6])

        return TradingSignal(
            signal_id=str(uuid7()),
            snapshot_id=snapshot.snapshot_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            market=snapshot.market,
            action=action,
            weight=weight,
            confidence=confidence,
            reasoning=reasoning or "Multi-agent pipeline completed",
            model=self._model_provider,
            architecture=AgentArchitecture.MULTI,
        )
