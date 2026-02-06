"""Tests for LLM response parser."""

from __future__ import annotations

import pytest

from src.core.types import Action, AgentArchitecture, Market, ModelProvider
from src.llm.response_parser import ResponseParser


@pytest.fixture
def parser() -> ResponseParser:
    return ResponseParser()


class TestJSONParsing:
    def test_valid_json(self, parser: ResponseParser) -> None:
        raw = '{"action": "BUY", "weight": 0.15, "confidence": 0.8, "reasoning": "Strong momentum"}'
        signal, ok = parser.parse_signal(
            raw, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE, "snap-1", "BTCUSDT", Market.CRYPTO
        )
        assert ok is True
        assert signal.action == Action.BUY
        assert signal.weight == 0.15
        assert signal.confidence == 0.8

    def test_json_in_code_block(self, parser: ResponseParser) -> None:
        raw = '```json\n{"action": "SELL", "weight": 0.1, "confidence": 0.6, "reasoning": "Bearish signal"}\n```'
        signal, ok = parser.parse_signal(
            raw, ModelProvider.GPT, AgentArchitecture.SINGLE, "snap-1", "AAPL", Market.US
        )
        assert ok is True
        assert signal.action == Action.SELL

    def test_invalid_json_falls_to_regex(self, parser: ResponseParser) -> None:
        raw = "I recommend BUY with weight 0.2 and confidence 0.7 because strong fundamentals"
        signal, ok = parser.parse_signal(
            raw, ModelProvider.GEMINI, AgentArchitecture.SINGLE, "snap-1", "005930", Market.KRX
        )
        assert ok is True
        assert signal.action == Action.BUY


class TestXMLParsing:
    def test_valid_xml(self, parser: ResponseParser) -> None:
        raw = """<trading_signal>
<action>HOLD</action>
<weight>0.0</weight>
<confidence>0.5</confidence>
<reasoning>Market is sideways with no clear direction.</reasoning>
</trading_signal>"""
        signal, ok = parser.parse_signal(
            raw, ModelProvider.CLAUDE, AgentArchitecture.SINGLE, "snap-1", "ETHUSDT", Market.CRYPTO
        )
        assert ok is True
        assert signal.action == Action.HOLD
        assert signal.confidence == 0.5

    def test_xml_with_surrounding_text(self, parser: ResponseParser) -> None:
        raw = """Let me analyze the market.
<action>BUY</action>
<weight>0.25</weight>
<confidence>0.85</confidence>
<reasoning>Strong bullish divergence on RSI.</reasoning>
"""
        signal, ok = parser.parse_signal(
            raw, ModelProvider.CLAUDE, AgentArchitecture.MULTI, "snap-1", "BTCUSDT", Market.CRYPTO
        )
        assert ok is True
        assert signal.action == Action.BUY
        assert signal.weight == 0.25


class TestFallback:
    def test_unparseable_returns_hold(self, parser: ResponseParser) -> None:
        raw = "This is completely unparseable gibberish with no trading signals."
        signal, ok = parser.parse_signal(
            raw, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE, "snap-1", "BTCUSDT", Market.CRYPTO
        )
        assert ok is False
        assert signal.action == Action.HOLD

    def test_weight_clamped(self, parser: ResponseParser) -> None:
        raw = '{"action": "BUY", "weight": 5.0, "confidence": 2.0, "reasoning": "Extreme values"}'
        signal, ok = parser.parse_signal(
            raw, ModelProvider.GPT, AgentArchitecture.SINGLE, "snap-1", "AAPL", Market.US
        )
        assert ok is True
        assert signal.weight == 1.0
        assert signal.confidence == 1.0
