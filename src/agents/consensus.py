"""Multi-model consensus — ensemble signals from 4 LLMs for higher accuracy.

STATUS: NOT YET INTEGRATED — Consensus engine is implemented but not called
from the benchmark pipeline. Ready for cross-model ensemble signal aggregation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from uuid_extensions import uuid7

from src.core.logging import get_logger
from src.core.types import Action, ModelProvider, TradingSignal

log = get_logger(__name__)


# ── Consensus Tuning Constants ───────────────────────────────────
_UNANIMOUS_BOOST: float = 1.2
_CONFIDENCE_CAP: float = 1.0
_SPLIT_CONFIDENCE: float = 0.2
_SUSPICIOUS_CONFIDENCE_THRESHOLD: float = 0.7
_OUTLIER_CONFIDENCE_DEVIATION: float = 0.3


@dataclass
class OutlierFlag:
    """Flag raised when a model's signal deviates significantly from peers."""

    flag_id: str = field(default_factory=lambda: str(uuid7()))
    model: ModelProvider = ModelProvider.DEEPSEEK
    reason: str = ""
    severity: str = "info"  # "info" | "warning" | "suspicious"


@dataclass
class ConsensusSignal:
    """Ensemble signal built from multiple LLM signals."""

    consensus_id: str = field(default_factory=lambda: str(uuid7()))
    action: Action = Action.HOLD
    confidence: float = 0.0
    agreement_ratio: float = 0.0  # 0.25 ~ 1.0
    contributing_models: list[ModelProvider] = field(default_factory=list)
    dissenting_models: list[ModelProvider] = field(default_factory=list)
    reasoning: str = ""
    outlier_flags: list[OutlierFlag] = field(default_factory=list)


class ConsensusEngine:
    """Build consensus from multiple LLM signals.

    Rules:
    - 4/4 agree: confidence = avg * 1.2 (capped at 1.0)
    - 3/4 agree: confidence = avg * 1.0, dissenter flagged
    - 2/2 split: HOLD with low confidence
    - All different: HOLD

    Outlier detection:
    - If 3 say HOLD but 1 says strong BUY -> suspicious hallucination
    - If one model's confidence deviates strongly from the group -> flagged
    """

    def build_consensus(
        self,
        signals: dict[ModelProvider, TradingSignal],
    ) -> ConsensusSignal:
        """Build ensemble signal from individual model signals.

        Args:
            signals: Mapping of model provider to its trading signal.
                     Expects 2-4 entries for meaningful consensus.

        Returns:
            ConsensusSignal with the majority action, weighted confidence,
            and any outlier flags.
        """
        if not signals:
            log.warning("consensus_no_signals")
            return ConsensusSignal(
                action=Action.HOLD,
                confidence=0.0,
                agreement_ratio=0.0,
                reasoning="No signals provided for consensus.",
            )

        total_models: int = len(signals)

        # ── Count votes per action ────────────────────────────────
        action_votes: Counter[Action] = Counter()
        action_models: dict[Action, list[ModelProvider]] = {
            Action.BUY: [],
            Action.SELL: [],
            Action.HOLD: [],
        }

        for model, signal in signals.items():
            action_votes[signal.action] += 1
            action_models[signal.action].append(model)

        # ── Determine majority action ─────────────────────────────
        most_common: list[tuple[Action, int]] = action_votes.most_common()
        top_action: Action = most_common[0][0]
        top_count: int = most_common[0][1]

        # Check for true tie (e.g., 2 BUY vs 2 SELL)
        has_tie: bool = (
            len(most_common) >= 2 and most_common[0][1] == most_common[1][1]
        )

        # ── Detect outliers before building final signal ──────────
        outlier_flags: list[OutlierFlag] = self.detect_outliers(signals)

        # ── Build consensus based on agreement level ──────────────
        if has_tie:
            # Tied vote -> resolve to HOLD
            consensus: ConsensusSignal = self._build_tie_consensus(
                signals=signals,
                action_models=action_models,
                most_common=most_common,
                total_models=total_models,
                outlier_flags=outlier_flags,
            )
        elif top_count == total_models:
            # Unanimous agreement
            consensus = self._build_unanimous_consensus(
                signals=signals,
                action=top_action,
                action_models=action_models,
                total_models=total_models,
                outlier_flags=outlier_flags,
            )
        elif top_count >= 2:
            # Majority (e.g., 3/4 or 2/3)
            consensus = self._build_majority_consensus(
                signals=signals,
                action=top_action,
                action_models=action_models,
                top_count=top_count,
                total_models=total_models,
                outlier_flags=outlier_flags,
            )
        else:
            # All different actions (3-way split with 3+ models)
            consensus = self._build_no_consensus(
                signals=signals,
                action_models=action_models,
                total_models=total_models,
                outlier_flags=outlier_flags,
            )

        log.info(
            "consensus_built",
            action=consensus.action.value,
            confidence=f"{consensus.confidence:.2f}",
            agreement_ratio=f"{consensus.agreement_ratio:.2f}",
            contributing=len(consensus.contributing_models),
            dissenting=len(consensus.dissenting_models),
            outlier_flags=len(consensus.outlier_flags),
        )

        return consensus

    def detect_outliers(
        self,
        signals: dict[ModelProvider, TradingSignal],
    ) -> list[OutlierFlag]:
        """Detect models whose signals deviate significantly from peers.

        Checks for:
        1. Action outlier: a model disagrees when all others agree
        2. Confidence outlier: a model's confidence deviates far from group mean
        3. Hallucination suspect: strong directional signal when majority says HOLD

        Args:
            signals: Mapping of model provider to trading signal.

        Returns:
            List of OutlierFlag instances describing detected anomalies.
        """
        if len(signals) < 2:
            return []

        flags: list[OutlierFlag] = []
        total_models: int = len(signals)

        # ── Action Voting ────────────────────────────────────────
        action_votes: Counter[Action] = Counter(
            s.action for s in signals.values()
        )
        most_common_action: Action = action_votes.most_common(1)[0][0]
        most_common_count: int = action_votes[most_common_action]

        # ── Confidence Stats ─────────────────────────────────────
        confidences: list[float] = [s.confidence for s in signals.values()]
        avg_confidence: float = sum(confidences) / len(confidences)

        for model, signal in signals.items():
            # 1. Action outlier: single dissenter against unified majority
            if (
                signal.action != most_common_action
                and most_common_count >= total_models - 1
                and total_models >= 3
            ):
                # Hallucination check: strong directional vs group HOLD
                if (
                    most_common_action == Action.HOLD
                    and signal.action in (Action.BUY, Action.SELL)
                    and signal.confidence >= _SUSPICIOUS_CONFIDENCE_THRESHOLD
                ):
                    flags.append(OutlierFlag(
                        model=model,
                        reason=(
                            f"{model.value} signals strong {signal.action.value} "
                            f"(confidence={signal.confidence:.2f}) while "
                            f"{most_common_count} others signal HOLD. "
                            "Possible hallucination."
                        ),
                        severity="suspicious",
                    ))
                else:
                    flags.append(OutlierFlag(
                        model=model,
                        reason=(
                            f"{model.value} signals {signal.action.value} "
                            f"while {most_common_count} others signal "
                            f"{most_common_action.value}."
                        ),
                        severity="warning",
                    ))

            # 2. Confidence outlier: far from group mean
            confidence_delta: float = abs(signal.confidence - avg_confidence)
            if confidence_delta > _OUTLIER_CONFIDENCE_DEVIATION:
                direction: str = (
                    "higher" if signal.confidence > avg_confidence else "lower"
                )
                flags.append(OutlierFlag(
                    model=model,
                    reason=(
                        f"{model.value} confidence ({signal.confidence:.2f}) is "
                        f"significantly {direction} than group average "
                        f"({avg_confidence:.2f}), delta={confidence_delta:.2f}."
                    ),
                    severity="info",
                ))

            # 3. Contradictory strong conviction: BUY vs SELL with high confidence
            if signal.action == Action.BUY and signal.confidence > 0.8:
                strong_sellers: list[ModelProvider] = [
                    m for m, s in signals.items()
                    if s.action == Action.SELL and s.confidence > 0.8 and m != model
                ]
                if strong_sellers:
                    flags.append(OutlierFlag(
                        model=model,
                        reason=(
                            f"{model.value} strong BUY conflicts with "
                            f"strong SELL from {[m.value for m in strong_sellers]}. "
                            "Extreme disagreement signals high uncertainty."
                        ),
                        severity="warning",
                    ))

        if flags:
            log.info(
                "outliers_detected",
                flag_count=len(flags),
                models_flagged=[f.model.value for f in flags],
            )

        return flags

    def _build_unanimous_consensus(
        self,
        *,
        signals: dict[ModelProvider, TradingSignal],
        action: Action,
        action_models: dict[Action, list[ModelProvider]],
        total_models: int,
        outlier_flags: list[OutlierFlag],
    ) -> ConsensusSignal:
        """All models agree on the same action."""
        contributing: list[ModelProvider] = action_models[action]
        avg_conf: float = self._weighted_confidence(signals, contributing)
        boosted_conf: float = min(avg_conf * _UNANIMOUS_BOOST, _CONFIDENCE_CAP)

        reasoning_parts: list[str] = [
            f"UNANIMOUS ({total_models}/{total_models}): All models agree on {action.value}.",
        ]
        reasoning_parts.extend(self._collect_reasoning_summary(signals, contributing))

        return ConsensusSignal(
            action=action,
            confidence=round(boosted_conf, 4),
            agreement_ratio=1.0,
            contributing_models=contributing,
            dissenting_models=[],
            reasoning=" ".join(reasoning_parts),
            outlier_flags=outlier_flags,
        )

    def _build_majority_consensus(
        self,
        *,
        signals: dict[ModelProvider, TradingSignal],
        action: Action,
        action_models: dict[Action, list[ModelProvider]],
        top_count: int,
        total_models: int,
        outlier_flags: list[OutlierFlag],
    ) -> ConsensusSignal:
        """Majority of models agree (e.g., 3/4)."""
        contributing: list[ModelProvider] = action_models[action]
        dissenters: list[ModelProvider] = [
            m for m in signals if m not in contributing
        ]
        avg_conf: float = self._weighted_confidence(signals, contributing)
        agreement_ratio: float = top_count / total_models

        reasoning_parts: list[str] = [
            f"MAJORITY ({top_count}/{total_models}): "
            f"{action.value} with {[m.value for m in contributing]} agreeing.",
        ]
        if dissenters:
            for d in dissenters:
                d_signal: TradingSignal = signals[d]
                reasoning_parts.append(
                    f"Dissenter {d.value}: {d_signal.action.value} "
                    f"(conf={d_signal.confidence:.2f})."
                )
        reasoning_parts.extend(self._collect_reasoning_summary(signals, contributing))

        return ConsensusSignal(
            action=action,
            confidence=round(avg_conf, 4),
            agreement_ratio=round(agreement_ratio, 4),
            contributing_models=contributing,
            dissenting_models=dissenters,
            reasoning=" ".join(reasoning_parts),
            outlier_flags=outlier_flags,
        )

    def _build_tie_consensus(
        self,
        *,
        signals: dict[ModelProvider, TradingSignal],
        action_models: dict[Action, list[ModelProvider]],
        most_common: list[tuple[Action, int]],
        total_models: int,
        outlier_flags: list[OutlierFlag],
    ) -> ConsensusSignal:
        """Tied vote (e.g., 2 BUY vs 2 SELL) -> HOLD."""
        tied_actions: list[Action] = [
            a for a, c in most_common if c == most_common[0][1]
        ]
        all_models: list[ModelProvider] = list(signals.keys())

        # In a tie, use the average confidence of all signals, scaled down
        all_conf: float = sum(s.confidence for s in signals.values()) / total_models

        reasoning_parts: list[str] = [
            f"TIE: {[a.value for a in tied_actions]} tied at "
            f"{most_common[0][1]}/{total_models} each. Defaulting to HOLD.",
        ]
        for action in tied_actions:
            models: list[ModelProvider] = action_models[action]
            reasoning_parts.append(
                f"{action.value}: {[m.value for m in models]}."
            )

        return ConsensusSignal(
            action=Action.HOLD,
            confidence=round(min(all_conf * 0.5, _SPLIT_CONFIDENCE), 4),
            agreement_ratio=round(most_common[0][1] / total_models, 4),
            contributing_models=[],
            dissenting_models=all_models,
            reasoning=" ".join(reasoning_parts),
            outlier_flags=outlier_flags,
        )

    def _build_no_consensus(
        self,
        *,
        signals: dict[ModelProvider, TradingSignal],
        action_models: dict[Action, list[ModelProvider]],
        total_models: int,
        outlier_flags: list[OutlierFlag],
    ) -> ConsensusSignal:
        """No clear majority (all different actions) -> HOLD."""
        all_models: list[ModelProvider] = list(signals.keys())

        reasoning_parts: list[str] = [
            f"NO CONSENSUS: {total_models} models produced {len(action_models)} "
            "different actions. Defaulting to HOLD.",
        ]
        for action, models in action_models.items():
            if models:
                reasoning_parts.append(
                    f"{action.value}: {[m.value for m in models]}."
                )

        return ConsensusSignal(
            action=Action.HOLD,
            confidence=round(_SPLIT_CONFIDENCE * 0.5, 4),
            agreement_ratio=round(1.0 / total_models, 4),
            contributing_models=[],
            dissenting_models=all_models,
            reasoning=" ".join(reasoning_parts),
            outlier_flags=outlier_flags,
        )

    def _weighted_confidence(
        self,
        signals: dict[ModelProvider, TradingSignal],
        models: list[ModelProvider],
    ) -> float:
        """Calculate average confidence for the given subset of models.

        Args:
            signals: All model signals.
            models: Subset of models to average over.

        Returns:
            Average confidence, or 0.0 if no models provided.
        """
        if not models:
            return 0.0
        total: float = sum(signals[m].confidence for m in models)
        return total / len(models)

    def _collect_reasoning_summary(
        self,
        signals: dict[ModelProvider, TradingSignal],
        models: list[ModelProvider],
    ) -> list[str]:
        """Collect abbreviated reasoning from contributing models.

        Args:
            signals: All model signals.
            models: Models whose reasoning to collect.

        Returns:
            List of summary strings, one per model.
        """
        summaries: list[str] = []
        max_reason_len: int = 150
        for model in models:
            reason: str = signals[model].reasoning
            truncated: str = (
                reason[:max_reason_len] + "..."
                if len(reason) > max_reason_len
                else reason
            )
            summaries.append(f"[{model.value}] {truncated}")
        return summaries
