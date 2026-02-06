"""Confidence calibration analysis — measure how well LLM confidence matches reality.

A well-calibrated model should have:
- 70% confidence signals correct ~70% of the time
- 90% confidence signals correct ~90% of the time

This module bins signals by confidence level and compares predicted vs actual
accuracy, producing calibration curves and reliability diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.logging import get_logger
from src.core.types import Action, ModelProvider, AgentArchitecture

log = get_logger(__name__)


@dataclass
class CalibrationBin:
    """A single bin in the calibration curve."""
    bin_start: float
    bin_end: float
    bin_midpoint: float
    predicted_confidence: float  # average confidence in this bin
    actual_accuracy: float  # fraction of correct predictions
    count: int
    gap: float = 0.0  # |predicted - actual|

    def __post_init__(self) -> None:
        self.gap = abs(self.predicted_confidence - self.actual_accuracy)


@dataclass
class CalibrationResult:
    """Full calibration analysis for a model/architecture combination."""
    model: ModelProvider
    architecture: AgentArchitecture
    bins: list[CalibrationBin]
    expected_calibration_error: float  # ECE: weighted average of bin gaps
    max_calibration_error: float  # MCE: max bin gap
    overconfidence_ratio: float  # fraction of bins where confidence > accuracy
    total_predictions: int
    brier_score: float  # Mean squared error of probability estimates

    @property
    def is_well_calibrated(self) -> bool:
        """ECE below 0.05 is considered well-calibrated."""
        return self.expected_calibration_error < 0.05


@dataclass
class PredictionRecord:
    """Single prediction record for calibration analysis."""
    confidence: float
    was_correct: bool
    action: Action
    model: ModelProvider
    architecture: AgentArchitecture


class CalibrationAnalyzer:
    """Analyze confidence calibration across models and architectures.

    Tracks prediction outcomes and generates calibration reports.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self._n_bins: int = n_bins
        self._records: list[PredictionRecord] = []

    def record(
        self,
        confidence: float,
        was_correct: bool,
        action: Action,
        model: ModelProvider,
        architecture: AgentArchitecture,
    ) -> None:
        """Record a single prediction outcome."""
        self._records.append(PredictionRecord(
            confidence=confidence,
            was_correct=was_correct,
            action=action,
            model=model,
            architecture=architecture,
        ))

    def analyze(
        self,
        model: ModelProvider | None = None,
        architecture: AgentArchitecture | None = None,
    ) -> CalibrationResult:
        """Generate calibration analysis for a specific model/architecture.

        Args:
            model: Filter to specific model (None = all).
            architecture: Filter to specific architecture (None = all).

        Returns:
            CalibrationResult with bins and metrics.
        """
        # Filter records
        filtered = self._records
        if model is not None:
            filtered = [r for r in filtered if r.model == model]
        if architecture is not None:
            filtered = [r for r in filtered if r.architecture == architecture]

        if not filtered:
            return CalibrationResult(
                model=model or ModelProvider.DEEPSEEK,
                architecture=architecture or AgentArchitecture.SINGLE,
                bins=[],
                expected_calibration_error=0.0,
                max_calibration_error=0.0,
                overconfidence_ratio=0.0,
                total_predictions=0,
                brier_score=0.0,
            )

        # Create bins
        bin_edges = np.linspace(0.0, 1.0, self._n_bins + 1)
        bins: list[CalibrationBin] = []
        total = len(filtered)

        for i in range(self._n_bins):
            lo = float(bin_edges[i])
            hi = float(bin_edges[i + 1])
            mid = (lo + hi) / 2.0

            bin_records = [
                r for r in filtered if lo <= r.confidence < hi
            ]
            # Include upper boundary in last bin
            if i == self._n_bins - 1:
                bin_records = [
                    r for r in filtered if lo <= r.confidence <= hi
                ]

            if not bin_records:
                continue

            avg_conf = sum(r.confidence for r in bin_records) / len(bin_records)
            accuracy = sum(1 for r in bin_records if r.was_correct) / len(bin_records)

            bins.append(CalibrationBin(
                bin_start=lo,
                bin_end=hi,
                bin_midpoint=mid,
                predicted_confidence=avg_conf,
                actual_accuracy=accuracy,
                count=len(bin_records),
            ))

        # Calculate ECE (Expected Calibration Error)
        ece = sum(b.gap * b.count / total for b in bins) if total > 0 else 0.0

        # MCE (Maximum Calibration Error)
        mce = max((b.gap for b in bins), default=0.0)

        # Overconfidence ratio
        overconf = (
            sum(1 for b in bins if b.predicted_confidence > b.actual_accuracy)
            / len(bins)
            if bins
            else 0.0
        )

        # Brier score
        brier = (
            sum(
                (r.confidence - (1.0 if r.was_correct else 0.0)) ** 2
                for r in filtered
            )
            / total
            if total > 0
            else 0.0
        )

        result = CalibrationResult(
            model=model or ModelProvider.DEEPSEEK,
            architecture=architecture or AgentArchitecture.SINGLE,
            bins=bins,
            expected_calibration_error=ece,
            max_calibration_error=mce,
            overconfidence_ratio=overconf,
            total_predictions=total,
            brier_score=brier,
        )

        log.info(
            "calibration_analysis_complete",
            model=result.model.value if model else "all",
            architecture=result.architecture.value if architecture else "all",
            total_predictions=total,
            ece=round(ece, 4),
            mce=round(mce, 4),
            brier_score=round(brier, 4),
            well_calibrated=result.is_well_calibrated,
        )

        return result

    def analyze_all(self) -> dict[str, CalibrationResult]:
        """Generate calibration analysis for every model/architecture combination."""
        results: dict[str, CalibrationResult] = {}

        models = set(r.model for r in self._records)
        architectures = set(r.architecture for r in self._records)

        for model in models:
            for arch in architectures:
                key = f"{model.value}_{arch.value}"
                results[key] = self.analyze(model=model, architecture=arch)

        log.info("calibration_all_analyzed", combinations=len(results))
        return results

    @property
    def record_count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
