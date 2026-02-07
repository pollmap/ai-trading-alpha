"""Tests for multi-model consensus engine."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from uuid_extensions import uuid7

from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    ModelProvider,
    TradingSignal,
)
from src.agents.consensus import ConsensusEngine, ConsensusSignal


@pytest.fixture
def engine() -> ConsensusEngine:
    return ConsensusEngine()


def _make_signal(
    model: ModelProvider,
    action: Action,
    confidence: float = 0.7,
) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id="snap-1",
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        market=Market.CRYPTO,
        action=action,
        weight=0.15,
        confidence=confidence,
        reasoning=f"{model.value} analysis: {action.value} with confidence {confidence}",
        model=model,
        architecture=AgentArchitecture.SINGLE,
    )


class TestConsensusEngine:
    def test_unanimous_buy(self, engine: ConsensusEngine) -> None:
        signals = {
            ModelProvider.DEEPSEEK: _make_signal(ModelProvider.DEEPSEEK, Action.BUY, 0.8),
            ModelProvider.GEMINI: _make_signal(ModelProvider.GEMINI, Action.BUY, 0.75),
            ModelProvider.CLAUDE: _make_signal(ModelProvider.CLAUDE, Action.BUY, 0.85),
            ModelProvider.GPT: _make_signal(ModelProvider.GPT, Action.BUY, 0.7),
        }
        result = engine.build_consensus(signals)
        assert result.action == Action.BUY
        assert result.agreement_ratio == 1.0
        assert result.confidence > 0.7  # boosted
        assert len(result.dissenting_models) == 0

    def test_majority_3_of_4(self, engine: ConsensusEngine) -> None:
        signals = {
            ModelProvider.DEEPSEEK: _make_signal(ModelProvider.DEEPSEEK, Action.BUY, 0.8),
            ModelProvider.GEMINI: _make_signal(ModelProvider.GEMINI, Action.BUY, 0.75),
            ModelProvider.CLAUDE: _make_signal(ModelProvider.CLAUDE, Action.BUY, 0.85),
            ModelProvider.GPT: _make_signal(ModelProvider.GPT, Action.SELL, 0.6),
        }
        result = engine.build_consensus(signals)
        assert result.action == Action.BUY
        assert result.agreement_ratio == 0.75
        assert ModelProvider.GPT in result.dissenting_models

    def test_tie_defaults_hold(self, engine: ConsensusEngine) -> None:
        signals = {
            ModelProvider.DEEPSEEK: _make_signal(ModelProvider.DEEPSEEK, Action.BUY, 0.8),
            ModelProvider.GEMINI: _make_signal(ModelProvider.GEMINI, Action.BUY, 0.7),
            ModelProvider.CLAUDE: _make_signal(ModelProvider.CLAUDE, Action.SELL, 0.8),
            ModelProvider.GPT: _make_signal(ModelProvider.GPT, Action.SELL, 0.7),
        }
        result = engine.build_consensus(signals)
        assert result.action == Action.HOLD

    def test_no_consensus_all_different(self, engine: ConsensusEngine) -> None:
        signals = {
            ModelProvider.DEEPSEEK: _make_signal(ModelProvider.DEEPSEEK, Action.BUY, 0.5),
            ModelProvider.GEMINI: _make_signal(ModelProvider.GEMINI, Action.SELL, 0.5),
            ModelProvider.CLAUDE: _make_signal(ModelProvider.CLAUDE, Action.HOLD, 0.5),
        }
        result = engine.build_consensus(signals)
        assert result.action == Action.HOLD

    def test_empty_signals(self, engine: ConsensusEngine) -> None:
        result = engine.build_consensus({})
        assert result.action == Action.HOLD
        assert result.confidence == 0.0

    def test_outlier_detection_hallucination(self, engine: ConsensusEngine) -> None:
        signals = {
            ModelProvider.DEEPSEEK: _make_signal(ModelProvider.DEEPSEEK, Action.HOLD, 0.5),
            ModelProvider.GEMINI: _make_signal(ModelProvider.GEMINI, Action.HOLD, 0.5),
            ModelProvider.CLAUDE: _make_signal(ModelProvider.CLAUDE, Action.HOLD, 0.5),
            ModelProvider.GPT: _make_signal(ModelProvider.GPT, Action.BUY, 0.9),
        }
        flags = engine.detect_outliers(signals)
        assert len(flags) > 0
        suspicious = [f for f in flags if f.severity == "suspicious"]
        assert len(suspicious) > 0

    def test_consensus_signal_has_reasoning(self, engine: ConsensusEngine) -> None:
        signals = {
            ModelProvider.DEEPSEEK: _make_signal(ModelProvider.DEEPSEEK, Action.BUY, 0.8),
            ModelProvider.GEMINI: _make_signal(ModelProvider.GEMINI, Action.BUY, 0.75),
        }
        result = engine.build_consensus(signals)
        assert len(result.reasoning) > 0
