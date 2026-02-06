"""Prompt templates for single-agent trading."""

from __future__ import annotations

from src.core.types import Market, MarketSnapshot, ModelProvider, PortfolioState


def build_system_prompt(market: Market, output_format: str = "json") -> str:
    """Build the system prompt for single-agent trading.

    Args:
        market: Target market for market-specific nuances.
        output_format: "json" or "xml" depending on model.
    """
    market_context = _market_context(market)

    if output_format == "xml":
        format_instructions = _xml_format_instructions()
    else:
        format_instructions = _json_format_instructions()

    return f"""You are an expert trading agent specializing in the {market.value} market.

## Your Role
Analyze the provided market data and current portfolio to generate a trading signal.

## Trading Rules
- LONG ONLY: You can only BUY or HOLD. No short selling.
- Maximum position weight: 30% of total portfolio per symbol.
- Minimum cash reserve: 20% of total portfolio value.
- If uncertain, default to HOLD. Capital preservation is priority.

## Analysis Framework
Evaluate using ALL of these dimensions:
1. **Fundamentals**: Valuation (PER/PBR), earnings quality, market cap
2. **Technicals**: Price action, volume patterns, momentum
3. **Macro**: Interest rates, inflation, currency, policy impact
4. **Sentiment**: News tone, market fear/greed levels

{market_context}

## Output Format
{format_instructions}

## Example
{_example_output(output_format)}

IMPORTANT: Always provide substantive reasoning. Empty or trivial reasoning is not acceptable."""


def build_user_prompt(snapshot: MarketSnapshot, portfolio: PortfolioState) -> str:
    """Build the user prompt with current market data and portfolio state."""
    return f"""## Current Market Data

{snapshot.to_prompt_summary()}

## Current Portfolio

{portfolio.to_prompt_summary()}

## Task
Based on the market data and your current portfolio above, generate a trading signal.
Choose ONE symbol to act on (or HOLD if no action is warranted).
Provide your analysis and reasoning."""


def _market_context(market: Market) -> str:
    """Market-specific analysis emphasis."""
    if market == Market.KRX:
        return """## KRX Market Context
- Pay special attention to foreign/institutional investor flows (they drive Korean market direction)
- USD/KRW exchange rate strongly impacts export-heavy companies (Samsung, SK Hynix, Hyundai)
- Korean market hours: 09:00-15:30 KST
- Consider chaebols' cross-holding dynamics"""

    elif market == Market.US:
        return """## US Market Context
- Focus on sector rotation patterns and Federal Reserve policy signals
- Earnings season schedule matters: pre/post earnings volatility
- Consider mega-cap tech concentration risk (Magnificent 7)
- Treasury yield curve shape is a key macro signal"""

    elif market == Market.CRYPTO:
        return """## Crypto Market Context
- 24/7 market: consider global timezone effects on volume
- Fear & Greed Index is a strong contrarian indicator at extremes
- Bitcoin dominance affects altcoin performance
- Watch for large whale movements and exchange inflow/outflow
- Higher volatility = tighter risk management needed"""

    return ""


def _json_format_instructions() -> str:
    return """Respond with ONLY a JSON object (no markdown, no code blocks):
{
  "action": "BUY" | "SELL" | "HOLD",
  "symbol": "<ticker symbol>",
  "weight": <0.0 to 0.30>,
  "confidence": <0.0 to 1.0>,
  "reasoning": "<detailed analysis and rationale>"
}"""


def _xml_format_instructions() -> str:
    return """Respond with XML tags:
<trading_signal>
  <action>BUY | SELL | HOLD</action>
  <symbol>ticker symbol</symbol>
  <weight>0.0 to 0.30</weight>
  <confidence>0.0 to 1.0</confidence>
  <reasoning>detailed analysis and rationale</reasoning>
</trading_signal>"""


def _example_output(fmt: str) -> str:
    if fmt == "xml":
        return """<trading_signal>
  <action>BUY</action>
  <symbol>BTCUSDT</symbol>
  <weight>0.15</weight>
  <confidence>0.72</confidence>
  <reasoning>BTC shows bullish RSI divergence on 4H timeframe with Fear&Greed at 35 (fear zone). Historically, buying during fear with technical confirmation yields strong risk-adjusted returns. Current portfolio has 85% cash, providing ample room for position building.</reasoning>
</trading_signal>"""
    else:
        return """{
  "action": "BUY",
  "symbol": "BTCUSDT",
  "weight": 0.15,
  "confidence": 0.72,
  "reasoning": "BTC shows bullish RSI divergence on 4H timeframe with Fear&Greed at 35 (fear zone). Historically, buying during fear with technical confirmation yields strong risk-adjusted returns. Current portfolio has 85% cash, providing ample room for position building."
}"""
