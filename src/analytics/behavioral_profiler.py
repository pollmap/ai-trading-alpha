"""Behavioral profiling — analyze each model's trading personality."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.core.logging import get_logger
from src.core.types import TradingSignal

log = get_logger(__name__)


@dataclass
class BehavioralProfile:
    """Behavioral profile for a model-architecture combination."""

    model: str
    architecture: str
    action_distribution: dict[str, float]  # BUY/SELL/HOLD percentages
    avg_confidence: float
    confidence_std: float
    contrarian_score: float  # % of signals opposing majority
    avg_weight: float
    total_signals: int


class BehavioralProfiler:
    """Analyze trading behavior patterns per model."""

    def profile(
        self,
        signals: list[TradingSignal],
        model: str,
        architecture: str,
        all_signals: list[TradingSignal] | None = None,
    ) -> BehavioralProfile:
        """Create behavioral profile for a model-architecture combination.

        Args:
            signals: This model's signals.
            all_signals: All models' signals (for contrarian score calculation).
        """
        if not signals:
            return BehavioralProfile(
                model=model, architecture=architecture,
                action_distribution={"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0},
                avg_confidence=0.0, confidence_std=0.0,
                contrarian_score=0.0, avg_weight=0.0, total_signals=0,
            )

        # Action distribution
        actions = [s.action.value for s in signals]
        counter = Counter(actions)
        total = len(actions)
        distribution = {
            "BUY": round(counter.get("BUY", 0) / total, 4),
            "SELL": round(counter.get("SELL", 0) / total, 4),
            "HOLD": round(counter.get("HOLD", 0) / total, 4),
        }

        # Confidence analysis
        confidences = np.array([s.confidence for s in signals])
        avg_confidence = round(float(np.mean(confidences)), 4)
        confidence_std = round(float(np.std(confidences, ddof=1)), 4) if len(confidences) > 1 else 0.0

        # Average weight
        weights = [s.weight for s in signals if s.action.value != "HOLD"]
        avg_weight = round(float(np.mean(weights)), 4) if weights else 0.0

        # Contrarian score
        contrarian_score = self._calculate_contrarian_score(signals, all_signals or [])

        return BehavioralProfile(
            model=model,
            architecture=architecture,
            action_distribution=distribution,
            avg_confidence=avg_confidence,
            confidence_std=confidence_std,
            contrarian_score=contrarian_score,
            avg_weight=avg_weight,
            total_signals=total,
        )

    def _calculate_contrarian_score(
        self,
        model_signals: list[TradingSignal],
        all_signals: list[TradingSignal],
    ) -> float:
        """Calculate what % of this model's signals oppose the majority."""
        if not all_signals or not model_signals:
            return 0.0

        # Group all signals by snapshot_id
        snapshot_groups: dict[str, list[TradingSignal]] = {}
        for s in all_signals:
            snapshot_groups.setdefault(s.snapshot_id, []).append(s)

        contrarian_count = 0
        total_compared = 0

        for signal in model_signals:
            group = snapshot_groups.get(signal.snapshot_id, [])
            others = [s for s in group if s.model != signal.model]

            if not others:
                continue

            majority_action = Counter(s.action.value for s in others).most_common(1)[0][0]
            total_compared += 1

            if signal.action.value != majority_action:
                contrarian_count += 1

        return round(contrarian_count / total_compared, 4) if total_compared > 0 else 0.0

    def extract_reasoning_keywords(
        self,
        signals: list[TradingSignal],
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Extract top keywords from reasoning texts using TF-IDF."""
        if not signals:
            return []

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            texts = [s.reasoning for s in signals if s.reasoning]
            if not texts:
                return []

            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words="english",
                min_df=2,
                ngram_range=(1, 2),
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()

            # Average TF-IDF score per term
            avg_scores = tfidf_matrix.mean(axis=0).A1
            top_indices = avg_scores.argsort()[-top_n:][::-1]

            return [
                (feature_names[i], round(float(avg_scores[i]), 4))
                for i in top_indices
            ]

        except ImportError:
            log.warning("sklearn_not_available_for_keyword_extraction")
            return []
