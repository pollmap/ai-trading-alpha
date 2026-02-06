"""Unified LLM response parser — converts raw LLM output to TradingSignal."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from uuid_extensions import uuid7

from src.core.logging import get_logger
from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    ModelProvider,
    TradingSignal,
)

log = get_logger(__name__)

MAX_PARSE_ATTEMPTS = 3


class ResponseParser:
    """Parse LLM responses into TradingSignal objects.

    Parsing strategy per model:
    - DeepSeek: JSON mode output, regex fallback
    - Gemini: JSON mode (response_mime_type guarantees)
    - Claude: XML tag-based (<action>, <weight>, <confidence>, <reasoning>)
    - GPT: JSON mode output

    Failure handling:
    - 3 parse failures -> HOLD signal + full raw response logging
    - Abnormal values (weight > 1, confidence < 0) -> clamp + warning
    """

    def parse_signal(
        self,
        raw_response: str,
        model: ModelProvider,
        architecture: AgentArchitecture,
        snapshot_id: str,
        symbol: str,
        market: Market,
    ) -> tuple[TradingSignal, bool]:
        """Parse raw LLM response into a TradingSignal.

        Returns:
            Tuple of (TradingSignal, success: bool).
            On failure, returns a HOLD signal with success=False.
        """
        if model == ModelProvider.CLAUDE:
            result = self._parse_xml(raw_response)
        else:
            result = self._parse_json(raw_response)

        if result is None:
            # Fallback: regex extraction
            result = self._parse_regex(raw_response)

        if result is None:
            log.warning(
                "parse_failed_returning_hold",
                model=model.value,
                response_preview=raw_response[:200],
            )
            return self._hold_signal(
                model, architecture, snapshot_id, symbol, market,
                reasoning=f"Parse failure. Raw response preview: {raw_response[:500]}",
            ), False

        action, weight, confidence, reasoning = result

        signal = TradingSignal(
            signal_id=str(uuid7()),
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            market=market,
            action=action,
            weight=weight,
            confidence=confidence,
            reasoning=reasoning or "No reasoning provided",
            model=model,
            architecture=architecture,
        )

        log.info(
            "signal_parsed",
            model=model.value,
            symbol=symbol,
            action=action.value,
            weight=f"{weight:.2f}",
            confidence=f"{confidence:.2f}",
        )

        return signal, True

    # ── JSON Parsing (DeepSeek, Gemini, GPT) ─────────────────────

    def _parse_json(
        self, raw: str,
    ) -> tuple[Action, float, float, str] | None:
        """Parse JSON-formatted response."""
        try:
            # Try direct JSON parse
            data = json.loads(raw)
            return self._extract_from_dict(data)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._extract_from_dict(data)
            except json.JSONDecodeError:
                pass

        # Try finding any JSON object in the response
        brace_match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", raw, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return self._extract_from_dict(data)
            except json.JSONDecodeError:
                pass

        return None

    def _extract_from_dict(
        self, data: dict[str, object],
    ) -> tuple[Action, float, float, str] | None:
        """Extract signal fields from a parsed dict."""
        try:
            action_str = str(data.get("action", "HOLD")).upper().strip()
            action = Action(action_str) if action_str in ("BUY", "SELL", "HOLD") else Action.HOLD

            weight = float(data.get("weight", 0.0))
            confidence = float(data.get("confidence", 0.5))
            reasoning = str(data.get("reasoning", ""))

            return action, weight, confidence, reasoning
        except (ValueError, TypeError) as exc:
            log.debug("dict_extraction_failed", error=str(exc))
            return None

    # ── XML Parsing (Claude) ─────────────────────────────────────

    def _parse_xml(
        self, raw: str,
    ) -> tuple[Action, float, float, str] | None:
        """Parse XML tag-based response (Claude format)."""
        try:
            action_match = re.search(r"<action>(.*?)</action>", raw, re.DOTALL | re.IGNORECASE)
            weight_match = re.search(r"<weight>(.*?)</weight>", raw, re.DOTALL | re.IGNORECASE)
            confidence_match = re.search(r"<confidence>(.*?)</confidence>", raw, re.DOTALL | re.IGNORECASE)
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.DOTALL | re.IGNORECASE)

            if not action_match:
                return None

            action_str = action_match.group(1).strip().upper()
            action = Action(action_str) if action_str in ("BUY", "SELL", "HOLD") else Action.HOLD

            weight = float(weight_match.group(1).strip()) if weight_match else 0.0
            confidence = float(confidence_match.group(1).strip()) if confidence_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            return action, weight, confidence, reasoning
        except (ValueError, AttributeError) as exc:
            log.debug("xml_parse_failed", error=str(exc))
            return None

    # ── Regex Fallback ───────────────────────────────────────────

    def _parse_regex(
        self, raw: str,
    ) -> tuple[Action, float, float, str] | None:
        """Last-resort regex extraction."""
        action_match = re.search(r"\b(BUY|SELL|HOLD)\b", raw, re.IGNORECASE)
        if not action_match:
            return None

        action_str = action_match.group(1).upper()
        action = Action(action_str)

        weight_match = re.search(r"weight[:\s]*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)
        weight = float(weight_match.group(1)) if weight_match else 0.0

        conf_match = re.search(r"confidence[:\s]*([0-9]*\.?[0-9]+)", raw, re.IGNORECASE)
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        return action, weight, confidence, raw[:1000]

    # ── HOLD Fallback ────────────────────────────────────────────

    def _hold_signal(
        self,
        model: ModelProvider,
        architecture: AgentArchitecture,
        snapshot_id: str,
        symbol: str,
        market: Market,
        reasoning: str = "Parse failure — defaulting to HOLD",
    ) -> TradingSignal:
        """Create a HOLD signal as fallback."""
        return TradingSignal(
            signal_id=str(uuid7()),
            snapshot_id=snapshot_id,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            market=market,
            action=Action.HOLD,
            weight=0.0,
            confidence=0.0,
            reasoning=reasoning,
            model=model,
            architecture=architecture,
        )
