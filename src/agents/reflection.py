"""Agent self-reflection — generate performance review for prompt injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid_extensions import uuid7

from src.core.logging import get_logger
from src.core.types import Action, TradingSignal

log = get_logger(__name__)


# ── Thresholds for Observation Generation ────────────────────────
_LOW_ACCURACY_THRESHOLD: float = 0.40
_HIGH_ACCURACY_THRESHOLD: float = 0.70
_OVERTRADING_HOLD_RATIO_THRESHOLD: float = 0.15
_CONFIDENCE_CALIBRATION_GAP: float = 0.20
_STREAK_WARNING_LENGTH: int = 3
_MIN_OUTCOMES_FOR_REFLECTION: int = 3


@dataclass
class TradeOutcome:
    """Record of a single trade's result for reflection analysis."""

    signal: TradingSignal = field(repr=False)
    entry_price: float = 0.0
    current_price: float = 0.0
    realized_pnl: float = 0.0
    holding_periods: int = 0
    was_correct: bool = False  # direction was right
    outcome_id: str = field(default_factory=lambda: str(uuid7()))


class AgentReflector:
    """Analyze recent trades and generate self-reflection text for prompt injection.

    The reflection text is appended to the system prompt, making agents
    aware of their recent performance patterns.
    """

    def __init__(self, lookback: int = 10) -> None:
        self._lookback: int = lookback
        self._history: dict[str, list[TradeOutcome]] = {}  # model_arch_key -> outcomes

    def record_outcome(self, key: str, outcome: TradeOutcome) -> None:
        """Record a trade outcome for future reflection.

        Args:
            key: Unique key identifying the model+architecture combination
                 (e.g. "deepseek_single_KRX").
            outcome: The completed trade outcome to record.
        """
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(outcome)
        # Prune to keep memory bounded (2x lookback, then trim)
        if len(self._history[key]) > self._lookback * 2:
            self._history[key] = self._history[key][-self._lookback:]
        log.debug(
            "reflection_outcome_recorded",
            key=key,
            was_correct=outcome.was_correct,
            realized_pnl=outcome.realized_pnl,
            history_size=len(self._history[key]),
        )

    def generate_reflection(self, key: str) -> str:
        """Generate natural language performance review.

        Returns empty string if insufficient history (fewer than 3 outcomes).

        Args:
            key: The model+architecture combination key.

        Returns:
            Multi-line reflection text suitable for prompt injection,
            or empty string if history is too short.
        """
        outcomes: list[TradeOutcome] = self._history.get(key, [])[-self._lookback:]
        if len(outcomes) < _MIN_OUTCOMES_FOR_REFLECTION:
            log.debug(
                "reflection_skipped_insufficient_history",
                key=key,
                outcome_count=len(outcomes),
                minimum_required=_MIN_OUTCOMES_FOR_REFLECTION,
            )
            return ""

        # ── Calculate Core Stats ──────────────────────────────────
        wins: int = sum(1 for o in outcomes if o.was_correct)
        total: int = len(outcomes)
        win_rate: float = wins / total

        buy_outcomes: list[TradeOutcome] = [
            o for o in outcomes if o.signal.action == Action.BUY
        ]
        sell_outcomes: list[TradeOutcome] = [
            o for o in outcomes if o.signal.action == Action.SELL
        ]
        hold_outcomes: list[TradeOutcome] = [
            o for o in outcomes if o.signal.action == Action.HOLD
        ]

        buy_accuracy: float = (
            sum(1 for o in buy_outcomes if o.was_correct) / len(buy_outcomes)
            if buy_outcomes
            else 0.0
        )
        sell_accuracy: float = (
            sum(1 for o in sell_outcomes if o.was_correct) / len(sell_outcomes)
            if sell_outcomes
            else 0.0
        )

        # PnL statistics
        winning_trades: list[TradeOutcome] = [o for o in outcomes if o.realized_pnl > 0]
        losing_trades: list[TradeOutcome] = [o for o in outcomes if o.realized_pnl < 0]

        avg_win_pnl: float = (
            sum(o.realized_pnl for o in winning_trades) / len(winning_trades)
            if winning_trades
            else 0.0
        )
        avg_loss_pnl: float = (
            sum(o.realized_pnl for o in losing_trades) / len(losing_trades)
            if losing_trades
            else 0.0
        )

        total_pnl: float = sum(o.realized_pnl for o in outcomes)
        avg_holding: float = (
            sum(o.holding_periods for o in outcomes) / total if total > 0 else 0.0
        )

        # Confidence calibration
        avg_confidence: float = (
            sum(o.signal.confidence for o in outcomes) / total if total > 0 else 0.0
        )

        # ── Build Reflection Text ─────────────────────────────────
        lines: list[str] = ["=== Self-Reflection (Recent Performance) ==="]
        lines.append(
            f"Last {total} trades: {wins}W / {total - wins}L "
            f"(Win Rate: {win_rate:.0%})"
        )
        lines.append(
            f"BUY accuracy: {buy_accuracy:.0%} ({len(buy_outcomes)} trades) | "
            f"SELL accuracy: {sell_accuracy:.0%} ({len(sell_outcomes)} trades)"
        )
        lines.append(
            f"Avg winning PnL: {avg_win_pnl:+.2f} | "
            f"Avg losing PnL: {avg_loss_pnl:+.2f} | "
            f"Net PnL: {total_pnl:+.2f}"
        )
        lines.append(f"Avg holding period: {avg_holding:.1f} cycles")

        # ── Generate Specific Observations ────────────────────────
        observations: list[str] = self._generate_observations(
            outcomes=outcomes,
            win_rate=win_rate,
            buy_accuracy=buy_accuracy,
            sell_accuracy=sell_accuracy,
            buy_count=len(buy_outcomes),
            sell_count=len(sell_outcomes),
            hold_count=len(hold_outcomes),
            avg_win_pnl=avg_win_pnl,
            avg_loss_pnl=avg_loss_pnl,
            avg_confidence=avg_confidence,
            avg_holding=avg_holding,
            total=total,
        )

        if observations:
            lines.append("")
            lines.append("Observations:")
            for obs in observations:
                lines.append(f"- {obs}")

        reflection_text: str = "\n".join(lines)
        log.info(
            "reflection_generated",
            key=key,
            win_rate=f"{win_rate:.2f}",
            total_trades=total,
            observation_count=len(observations),
        )
        return reflection_text

    def _generate_observations(
        self,
        *,
        outcomes: list[TradeOutcome],
        win_rate: float,
        buy_accuracy: float,
        sell_accuracy: float,
        buy_count: int,
        sell_count: int,
        hold_count: int,
        avg_win_pnl: float,
        avg_loss_pnl: float,
        avg_confidence: float,
        avg_holding: float,
        total: int,
    ) -> list[str]:
        """Generate actionable observations based on performance patterns.

        Returns:
            List of observation strings to include in the reflection.
        """
        observations: list[str] = []

        # ── Overtrading Warning ───────────────────────────────────
        hold_ratio: float = hold_count / total if total > 0 else 0.0
        if hold_ratio < _OVERTRADING_HOLD_RATIO_THRESHOLD:
            observations.append(
                f"WARNING: You rarely choose HOLD ({hold_ratio:.0%} of signals). "
                "Consider whether every cycle truly warrants a position change. "
                "Sometimes the best trade is no trade."
            )

        # ── Directional Bias ─────────────────────────────────────
        if buy_count > 0 and sell_count > 0:
            buy_ratio: float = buy_count / (buy_count + sell_count)
            if buy_ratio > 0.75:
                observations.append(
                    f"You show a strong BUY bias ({buy_ratio:.0%} of non-HOLD signals are BUY). "
                    "Verify you are not overlooking bearish signals."
                )
            elif buy_ratio < 0.25:
                observations.append(
                    f"You show a strong SELL bias ({1 - buy_ratio:.0%} of non-HOLD signals are SELL). "
                    "Verify you are not overlooking bullish opportunities."
                )

        # ── BUY Accuracy Problems ─────────────────────────────────
        if buy_count >= 2 and buy_accuracy < _LOW_ACCURACY_THRESHOLD:
            observations.append(
                f"Your BUY signals have been inaccurate ({buy_accuracy:.0%}). "
                "Consider raising your threshold for BUY conviction or "
                "reducing position size on BUY signals."
            )
        elif buy_count >= 2 and buy_accuracy > _HIGH_ACCURACY_THRESHOLD:
            observations.append(
                f"Your BUY timing has been excellent ({buy_accuracy:.0%} accuracy). "
                "Consider maintaining this approach."
            )

        # ── SELL Accuracy Problems ────────────────────────────────
        if sell_count >= 2 and sell_accuracy < _LOW_ACCURACY_THRESHOLD:
            observations.append(
                f"Your SELL signals have been inaccurate ({sell_accuracy:.0%}). "
                "Consider whether you are selling prematurely or "
                "misreading bearish signals."
            )
        elif sell_count >= 2 and sell_accuracy > _HIGH_ACCURACY_THRESHOLD:
            observations.append(
                f"Your SELL timing has been excellent ({sell_accuracy:.0%} accuracy). "
                "Continue trusting your bearish analysis."
            )

        # ── Confidence Calibration ────────────────────────────────
        calibration_gap: float = abs(avg_confidence - win_rate)
        if calibration_gap > _CONFIDENCE_CALIBRATION_GAP:
            if avg_confidence > win_rate:
                observations.append(
                    f"CALIBRATION: Your average confidence ({avg_confidence:.0%}) "
                    f"exceeds your actual win rate ({win_rate:.0%}). "
                    "You are overconfident. Lower your confidence scores or "
                    "be more selective with high-confidence signals."
                )
            else:
                observations.append(
                    f"CALIBRATION: Your average confidence ({avg_confidence:.0%}) "
                    f"is below your actual win rate ({win_rate:.0%}). "
                    "You are underconfident. You may be missing opportunities "
                    "by assigning low confidence to good signals."
                )

        # ── Risk/Reward Ratio ─────────────────────────────────────
        if avg_loss_pnl < 0 and avg_win_pnl > 0:
            risk_reward: float = avg_win_pnl / abs(avg_loss_pnl)
            if risk_reward < 0.8:
                observations.append(
                    f"Poor risk/reward ratio ({risk_reward:.2f}x). "
                    "Your average loss exceeds your average win. "
                    "Consider tighter stop-losses or larger profit targets."
                )
            elif risk_reward > 2.0:
                observations.append(
                    f"Excellent risk/reward ratio ({risk_reward:.2f}x). "
                    "Your winners significantly outpace your losers."
                )

        # ── Losing Streak Detection ──────────────────────────────
        current_streak: int = 0
        max_losing_streak: int = 0
        for outcome in outcomes:
            if not outcome.was_correct:
                current_streak += 1
                max_losing_streak = max(max_losing_streak, current_streak)
            else:
                current_streak = 0

        if max_losing_streak >= _STREAK_WARNING_LENGTH:
            observations.append(
                f"You experienced a {max_losing_streak}-trade losing streak recently. "
                "Consider reducing position sizes until confidence rebuilds."
            )

        # ── Recent Trend (last 3 vs earlier) ─────────────────────
        if len(outcomes) >= 6:
            recent: list[TradeOutcome] = outcomes[-3:]
            earlier: list[TradeOutcome] = outcomes[:-3]
            recent_wr: float = sum(1 for o in recent if o.was_correct) / len(recent)
            earlier_wr: float = (
                sum(1 for o in earlier if o.was_correct) / len(earlier)
                if earlier
                else 0.0
            )
            delta: float = recent_wr - earlier_wr
            if delta > 0.25:
                observations.append(
                    f"IMPROVING: Recent win rate ({recent_wr:.0%}) is significantly "
                    f"above earlier performance ({earlier_wr:.0%}). "
                    "Your recent adjustments appear effective."
                )
            elif delta < -0.25:
                observations.append(
                    f"DETERIORATING: Recent win rate ({recent_wr:.0%}) has dropped "
                    f"compared to earlier ({earlier_wr:.0%}). "
                    "The market regime may have changed. Reassess your strategy."
                )

        # ── Holding Period Analysis ──────────────────────────────
        if avg_holding < 1.5 and total >= 5:
            observations.append(
                f"Very short average holding period ({avg_holding:.1f} cycles). "
                "You may be over-reacting to short-term noise. "
                "Consider letting positions develop longer."
            )

        return observations

    def clear_history(self, key: str) -> None:
        """Clear all recorded outcomes for a given key.

        Args:
            key: The model+architecture combination key to clear.
        """
        if key in self._history:
            del self._history[key]
            log.info("reflection_history_cleared", key=key)

    def get_outcome_count(self, key: str) -> int:
        """Return the number of recorded outcomes for a key.

        Args:
            key: The model+architecture combination key.

        Returns:
            Number of outcomes currently stored.
        """
        return len(self._history.get(key, []))
