"""Tests for hallucination detection."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from uuid_extensions import uuid7

from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    MarketSnapshot,
    MacroData,
    ModelProvider,
    SymbolData,
    TradingSignal,
)
from src.llm.hallucination_detector import (
    FlaggedClaim,
    HallucinationDetector,
    HallucinationReport,
)


@pytest.fixture
def detector() -> HallucinationDetector:
    return HallucinationDetector(price_tolerance=0.05)


@pytest.fixture
def snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id="snap-test",
        timestamp=datetime.now(timezone.utc),
        market=Market.US,
        symbols={
            "AAPL": SymbolData(
                symbol="AAPL", market=Market.US,
                open=150.0, high=155.0, low=149.0, close=152.0,
                volume=50_000_000.0, currency="USD",
            ),
            "GOOG": SymbolData(
                symbol="GOOG", market=Market.US,
                open=140.0, high=142.0, low=138.0, close=139.0,
                volume=30_000_000.0, currency="USD",
            ),
        },
        macro=MacroData(),
    )


def _signal(action: Action = Action.BUY) -> TradingSignal:
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id="snap-test",
        timestamp=datetime.now(timezone.utc),
        symbol="AAPL",
        market=Market.US,
        action=action,
        weight=0.15,
        confidence=0.8,
        reasoning="Test reasoning",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
    )


class TestHallucinationDetector:
    def test_clean_reasoning(self, detector: HallucinationDetector, snapshot: MarketSnapshot) -> None:
        reasoning = "AAPL is showing strength at price $152 with good volume."
        report = detector.validate(_signal(), snapshot, reasoning)
        assert report.is_clean
        assert report.recommendation == "proceed"

    def test_ticker_hallucination(self, detector: HallucinationDetector, snapshot: MarketSnapshot) -> None:
        reasoning = "NVDA is surging while AAPL shows sideways movement"
        report = detector.validate(_signal(), snapshot, reasoning)
        # NVDA not in snapshot - should flag
        assert any(f.claim_type == "ticker" for f in report.flagged_claims)

    def test_price_hallucination(self, detector: HallucinationDetector, snapshot: MarketSnapshot) -> None:
        reasoning = "AAPL trading at $200 is significantly overvalued"
        report = detector.validate(_signal(), snapshot, reasoning)
        # $200 is far from actual $152
        price_flags = [f for f in report.flagged_claims if f.claim_type == "price"]
        assert len(price_flags) > 0

    def test_direction_hallucination(self, detector: HallucinationDetector, snapshot: MarketSnapshot) -> None:
        # GOOG actually fell (open=140, close=139), but reasoning says rising
        reasoning = "GOOG is surging and rallying strongly today"
        signal = _signal()
        report = detector.validate(signal, snapshot, reasoning)
        direction_flags = [f for f in report.flagged_claims if f.claim_type == "direction"]
        assert len(direction_flags) > 0

    def test_penalty_calculation(self, detector: HallucinationDetector, snapshot: MarketSnapshot) -> None:
        # Multiple hallucinations should increase penalty
        reasoning = "NVDA at $500 is surging while TSLA crashes to $10"
        report = detector.validate(_signal(), snapshot, reasoning)
        assert report.confidence_penalty > 0
        assert report.confidence_penalty <= 1.0

    def test_empty_reasoning(self, detector: HallucinationDetector, snapshot: MarketSnapshot) -> None:
        report = detector.validate(_signal(), snapshot, "")
        assert report.is_clean  # nothing to flag

    def test_report_properties(self, detector: HallucinationDetector, snapshot: MarketSnapshot) -> None:
        report = HallucinationReport(
            is_clean=False,
            flagged_claims=[
                FlaggedClaim("test", "ticker", "expected", "actual", "critical"),
                FlaggedClaim("test2", "price", "expected2", "actual2", "minor"),
            ],
            confidence_penalty=0.6,
            recommendation="reject_to_hold",
        )
        assert report.critical_count == 1
