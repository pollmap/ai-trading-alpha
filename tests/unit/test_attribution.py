"""Tests for decision attribution tracker."""

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
from src.analytics.attribution import (
    AttributionTracker,
    DataSource,
    DecisionAttribution,
)


@pytest.fixture
def tracker() -> AttributionTracker:
    return AttributionTracker()


def _make_signal(
    action: Action = Action.BUY,
    reasoning: str = "Test reasoning",
) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id="snap-1",
        timestamp=datetime.now(timezone.utc),
        symbol="BTCUSDT",
        market=Market.CRYPTO,
        action=action,
        weight=0.15,
        confidence=0.8,
        reasoning=reasoning,
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
    )


class TestAttributionTracker:
    def test_empty_reasoning(self, tracker: AttributionTracker) -> None:
        signal = _make_signal(reasoning="No data references here at all")
        attr = tracker.attribute(signal)
        assert attr.signal_id == signal.signal_id
        assert attr.symbol == "BTCUSDT"

    def test_price_action_attribution(self, tracker: AttributionTracker) -> None:
        signal = _make_signal(
            reasoning="The price broke through resistance at support level with strong breakout"
        )
        attr = tracker.attribute(signal)
        sources = {a.source for a in attr.attributions}
        assert DataSource.PRICE_ACTION in sources

    def test_technical_attribution(self, tracker: AttributionTracker) -> None:
        signal = _make_signal(
            reasoning="RSI is oversold at 25, MACD showing bullish crossover, SMA trend positive"
        )
        attr = tracker.attribute(signal)
        sources = {a.source for a in attr.attributions}
        assert DataSource.TECHNICAL_INDICATORS in sources

    def test_macro_attribution(self, tracker: AttributionTracker) -> None:
        signal = _make_signal(
            reasoning="Fed rate decision expected, CPI inflation trending down"
        )
        attr = tracker.attribute(signal)
        sources = {a.source for a in attr.attributions}
        assert DataSource.MACRO_DATA in sources

    def test_multiple_sources(self, tracker: AttributionTracker) -> None:
        signal = _make_signal(
            reasoning="RSI oversold, price at support, Fed rate cut expected, "
            "news headlines positive, social sentiment bullish on reddit"
        )
        attr = tracker.attribute(signal)
        assert len(attr.attributions) >= 3
        # Weights should sum to ~1.0
        total_weight = sum(a.weight for a in attr.attributions)
        assert abs(total_weight - 1.0) < 0.01

    def test_dominant_source(self, tracker: AttributionTracker) -> None:
        signal = _make_signal(
            reasoning="RSI MACD SMA EMA bollinger crossover momentum indicator oversold overbought"
        )
        attr = tracker.attribute(signal)
        assert attr.dominant_source == DataSource.TECHNICAL_INDICATORS

    def test_record_outcome_and_summarize(self, tracker: AttributionTracker) -> None:
        for i in range(10):
            signal = _make_signal(
                reasoning="RSI oversold, price at support level"
            )
            attr = tracker.attribute(signal)
            tracker.record_outcome(attr, 100.0 if i % 2 == 0 else -50.0)

        summary = tracker.summarize()
        assert summary.total_decisions == 10
        assert len(summary.source_win_rates) > 0
        assert len(summary.source_avg_weights) > 0

    def test_empty_summary(self, tracker: AttributionTracker) -> None:
        summary = tracker.summarize()
        assert summary.total_decisions == 0
