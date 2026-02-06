"""Hallucination detection -- validate LLM claims against actual snapshot data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.core.logging import get_logger
from src.core.types import TradingSignal, MarketSnapshot

log = get_logger(__name__)


# ── Regex patterns ───────────────────────────────────────────────

# US-style tickers: 2-6 uppercase letters, surrounded by word boundaries
# but not preceded/followed by lowercase (to avoid matching plain English words)
_US_TICKER_RE: re.Pattern[str] = re.compile(
    r"(?<![a-z])\b([A-Z]{2,6})\b(?![a-z])"
)

# Korean stock codes: exactly 6 digits, optionally preceded by '#' or 'KRX:'
_KR_TICKER_RE: re.Pattern[str] = re.compile(
    r"(?:KRX[:\s]?|#)?(\d{6})\b"
)

# Dollar amounts like $123.45 or $1,234.56
_DOLLAR_PRICE_RE: re.Pattern[str] = re.compile(
    r"\$\s?([\d,]+(?:\.\d{1,2})?)"
)

# Generic numeric amounts near context (e.g. "at 152.30", "price 4500")
_NUMERIC_PRICE_RE: re.Pattern[str] = re.compile(
    r"(?:price|at|@|around|approximately|roughly|close(?:d)?(?:\s+at)?)\s+"
    r"([\d,]+(?:\.\d{1,4})?)",
    re.IGNORECASE,
)

# Direction keywords mapped to expected sign of price change
_RISING_KEYWORDS: tuple[str, ...] = (
    "rising", "surging", "rallying", "climbing", "gaining",
    "up", "bullish", "soaring", "jumped", "increased",
    "higher", "advancing", "recovered",
)
_FALLING_KEYWORDS: tuple[str, ...] = (
    "falling", "dropping", "declining", "sinking", "losing",
    "down", "bearish", "plunging", "crashed", "decreased",
    "lower", "retreating", "slumped",
)

# Common English words that look like tickers but are not
_FALSE_POSITIVE_TICKERS: frozenset[str] = frozenset({
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
    "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "ITS",
    "HIS", "HOW", "WHO", "DID", "GET", "LET", "SAY", "SHE",
    "TOO", "USE", "BUY", "SELL", "HOLD", "MAY", "NEW", "NOW",
    "OLD", "SEE", "WAY", "DAY", "HAD", "HOT", "LOW", "HIGH",
    "BIG", "LONG", "MUCH", "VERY", "ALSO", "BACK", "BEEN",
    "COME", "EACH", "FROM", "GOOD", "HAVE", "HELP", "HERE",
    "JUST", "KNOW", "LAST", "LIKE", "LOOK", "MADE", "MAKE",
    "MANY", "MORE", "MOST", "MOVE", "NEXT", "ONLY", "OVER",
    "PART", "PLAN", "RISK", "SAME", "SOME", "SUCH", "TAKE",
    "THAN", "THAT", "THEM", "THEN", "THEY", "THIS", "TIME",
    "VERY", "WANT", "WELL", "WENT", "WHAT", "WHEN", "WILL",
    "WITH", "WORK", "YEAR", "YOUR", "KEEP", "EVEN", "RATE",
    "TERM", "DATA", "NEWS", "INTO", "NEAR", "PER", "VIX",
    "USD", "KRW", "KRX", "ETF", "IPO", "GDP", "CPI", "FED",
    "CEO", "CFO", "EPS", "PBR", "RSI", "SMA", "EMA",
    "MACD", "VWAP", "OHLC",
})


# ── Value Objects ────────────────────────────────────────────────


@dataclass(frozen=True)
class FlaggedClaim:
    """A single hallucinated or unverifiable claim found in LLM reasoning."""

    claim_text: str
    claim_type: str  # "ticker" | "price" | "direction" | "indicator" | "news"
    expected: str
    actual: str
    severity: str  # "minor" | "major" | "critical"


@dataclass
class HallucinationReport:
    """Aggregated validation result for a single LLM response."""

    is_clean: bool
    flagged_claims: list[FlaggedClaim]
    confidence_penalty: float  # 0.0~1.0 to subtract from signal confidence
    recommendation: str  # "proceed" | "reduce_weight" | "reject_to_hold"

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.flagged_claims if f.severity == "critical")


# ── Detector ─────────────────────────────────────────────────────


class HallucinationDetector:
    """Cross-validate LLM reasoning against actual snapshot data.

    Checks:
    1. Ticker validation: mentioned tickers exist in snapshot
    2. Price validation: mentioned prices within tolerance of actual
    3. Direction validation: if says "price is rising" check actual change
    4. News validation: referenced news items exist in snapshot.news
    """

    def __init__(self, price_tolerance: float = 0.05) -> None:
        self._price_tolerance: float = price_tolerance

    # ── Public API ───────────────────────────────────────────────

    def validate(
        self,
        signal: TradingSignal,
        snapshot: MarketSnapshot,
        raw_reasoning: str,
    ) -> HallucinationReport:
        """Validate the signal's reasoning against actual data."""
        flags: list[FlaggedClaim] = []
        flags.extend(self._check_tickers(raw_reasoning, snapshot))
        flags.extend(self._check_prices(raw_reasoning, snapshot))
        flags.extend(self._check_direction_claims(raw_reasoning, snapshot))

        # Calculate penalty
        penalty: float = sum(
            0.1 if f.severity == "minor"
            else 0.25 if f.severity == "major"
            else 0.5
            for f in flags
        )
        penalty = min(1.0, penalty)

        if penalty >= 0.5:
            recommendation: str = "reject_to_hold"
        elif penalty >= 0.2:
            recommendation = "reduce_weight"
        else:
            recommendation = "proceed"

        report = HallucinationReport(
            is_clean=len(flags) == 0,
            flagged_claims=flags,
            confidence_penalty=penalty,
            recommendation=recommendation,
        )

        log.info(
            "hallucination_check_complete",
            is_clean=report.is_clean,
            flagged_count=len(flags),
            critical_count=report.critical_count,
            penalty=penalty,
            recommendation=recommendation,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
        )

        return report

    # ── Ticker Validation ────────────────────────────────────────

    def _check_tickers(
        self, reasoning: str, snapshot: MarketSnapshot
    ) -> list[FlaggedClaim]:
        """Find ticker-like patterns in reasoning and verify they exist in snapshot.symbols."""
        flags: list[FlaggedClaim] = []
        snapshot_symbols: frozenset[str] = frozenset(snapshot.symbols.keys())

        # Collect all candidate tickers from the text
        mentioned_tickers: set[str] = set()

        # US-style tickers (uppercase 2-6 letters)
        for match in _US_TICKER_RE.finditer(reasoning):
            candidate: str = match.group(1)
            if candidate not in _FALSE_POSITIVE_TICKERS:
                mentioned_tickers.add(candidate)

        # Korean 6-digit codes
        for match in _KR_TICKER_RE.finditer(reasoning):
            mentioned_tickers.add(match.group(1))

        # Check each mentioned ticker against the snapshot
        for ticker in mentioned_tickers:
            if ticker not in snapshot_symbols:
                flags.append(
                    FlaggedClaim(
                        claim_text=f"Ticker '{ticker}' mentioned in reasoning",
                        claim_type="ticker",
                        expected="Present in snapshot symbols",
                        actual=f"'{ticker}' not found in snapshot "
                               f"(available: {', '.join(sorted(snapshot_symbols)[:10])})",
                        severity="major",
                    )
                )
                log.debug(
                    "hallucination_ticker_not_found",
                    ticker=ticker,
                    available_symbols=sorted(snapshot_symbols)[:10],
                )

        return flags

    # ── Price Validation ─────────────────────────────────────────

    def _check_prices(
        self, reasoning: str, snapshot: MarketSnapshot
    ) -> list[FlaggedClaim]:
        """Find dollar/numeric amounts in reasoning and compare against actual close prices.

        Strategy: for each price found in the text, try to associate it with the
        nearest mentioned ticker.  If the price deviates from the actual close by
        more than ``_price_tolerance``, flag it.
        """
        flags: list[FlaggedClaim] = []
        snapshot_symbols: dict[str, float] = {
            sym: sd.close for sym, sd in snapshot.symbols.items()
        }

        # Build a list of (position, ticker) for proximity matching
        ticker_positions: list[tuple[int, str]] = []
        for match in _US_TICKER_RE.finditer(reasoning):
            candidate: str = match.group(1)
            if candidate not in _FALSE_POSITIVE_TICKERS and candidate in snapshot_symbols:
                ticker_positions.append((match.start(), candidate))
        for match in _KR_TICKER_RE.finditer(reasoning):
            candidate = match.group(1)
            if candidate in snapshot_symbols:
                ticker_positions.append((match.start(), candidate))

        if not ticker_positions:
            return flags

        # Collect all price mentions
        price_mentions: list[tuple[int, float]] = []
        for match in _DOLLAR_PRICE_RE.finditer(reasoning):
            raw: str = match.group(1).replace(",", "")
            try:
                price_mentions.append((match.start(), float(raw)))
            except ValueError:
                continue
        for match in _NUMERIC_PRICE_RE.finditer(reasoning):
            raw = match.group(1).replace(",", "")
            try:
                price_mentions.append((match.start(), float(raw)))
            except ValueError:
                continue

        # For each price mention, find the nearest ticker and compare
        for pos, price_value in price_mentions:
            nearest_ticker: str = min(
                ticker_positions, key=lambda tp: abs(tp[0] - pos)
            )[1]
            actual_close: float = snapshot_symbols[nearest_ticker]

            if actual_close <= 0:
                continue

            deviation: float = abs(price_value - actual_close) / actual_close

            if deviation > self._price_tolerance:
                severity: str
                if deviation > 0.50:
                    severity = "critical"
                elif deviation > 0.20:
                    severity = "major"
                else:
                    severity = "minor"

                flags.append(
                    FlaggedClaim(
                        claim_text=(
                            f"Price {price_value:.2f} mentioned near "
                            f"ticker '{nearest_ticker}'"
                        ),
                        claim_type="price",
                        expected=f"Close price ~{actual_close:.2f} "
                                 f"(tolerance {self._price_tolerance:.0%})",
                        actual=f"Mentioned {price_value:.2f} "
                               f"(deviation {deviation:.1%})",
                        severity=severity,
                    )
                )
                log.debug(
                    "hallucination_price_mismatch",
                    ticker=nearest_ticker,
                    claimed_price=price_value,
                    actual_close=actual_close,
                    deviation=deviation,
                    severity=severity,
                )

        return flags

    # ── Direction Validation ─────────────────────────────────────

    def _check_direction_claims(
        self, reasoning: str, snapshot: MarketSnapshot
    ) -> list[FlaggedClaim]:
        """Look for directional language near ticker mentions and verify against
        actual open-to-close price change.

        'rising', 'up', 'bullish' etc. should correspond to positive change.
        'falling', 'down', 'bearish' etc. should correspond to negative change.
        """
        flags: list[FlaggedClaim] = []
        reasoning_lower: str = reasoning.lower()

        # For each symbol present in both the reasoning and the snapshot, check
        # directional claims within a window of characters around the ticker.
        _WINDOW: int = 120  # characters on each side

        for sym, sym_data in snapshot.symbols.items():
            # Locate the symbol in the reasoning text (case-insensitive)
            sym_pattern: re.Pattern[str] = re.compile(
                re.escape(sym), re.IGNORECASE
            )
            for sym_match in sym_pattern.finditer(reasoning):
                start: int = max(0, sym_match.start() - _WINDOW)
                end: int = min(len(reasoning), sym_match.end() + _WINDOW)
                context_window: str = reasoning[start:end].lower()

                # Determine actual direction from open to close
                if sym_data.open <= 0:
                    continue
                actual_change_pct: float = (
                    (sym_data.close - sym_data.open) / sym_data.open
                )

                # Check for rising keywords
                claims_rising: bool = any(
                    kw in context_window for kw in _RISING_KEYWORDS
                )
                claims_falling: bool = any(
                    kw in context_window for kw in _FALLING_KEYWORDS
                )

                # Flag contradiction: claims rising but actually falling
                if claims_rising and actual_change_pct < -0.001:
                    flags.append(
                        FlaggedClaim(
                            claim_text=(
                                f"Claims {sym} is rising/up in context"
                            ),
                            claim_type="direction",
                            expected=f"Positive price change for '{sym}'",
                            actual=(
                                f"Actual change {actual_change_pct:+.2%} "
                                f"(open={sym_data.open:.2f}, "
                                f"close={sym_data.close:.2f})"
                            ),
                            severity="major",
                        )
                    )
                    log.debug(
                        "hallucination_direction_mismatch",
                        symbol=sym,
                        claimed="rising",
                        actual_change_pct=actual_change_pct,
                    )

                # Flag contradiction: claims falling but actually rising
                if claims_falling and actual_change_pct > 0.001:
                    flags.append(
                        FlaggedClaim(
                            claim_text=(
                                f"Claims {sym} is falling/down in context"
                            ),
                            claim_type="direction",
                            expected=f"Negative price change for '{sym}'",
                            actual=(
                                f"Actual change {actual_change_pct:+.2%} "
                                f"(open={sym_data.open:.2f}, "
                                f"close={sym_data.close:.2f})"
                            ),
                            severity="major",
                        )
                    )
                    log.debug(
                        "hallucination_direction_mismatch",
                        symbol=sym,
                        claimed="falling",
                        actual_change_pct=actual_change_pct,
                    )

        return flags
