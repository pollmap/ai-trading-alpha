"""Tests for confidence calibration analysis."""

from __future__ import annotations

import pytest

from src.core.types import Action, AgentArchitecture, ModelProvider
from src.analytics.calibration import (
    CalibrationAnalyzer,
    CalibrationResult,
)


@pytest.fixture
def analyzer() -> CalibrationAnalyzer:
    return CalibrationAnalyzer(n_bins=5)


class TestCalibrationAnalyzer:
    def test_empty_analysis(self, analyzer: CalibrationAnalyzer) -> None:
        result = analyzer.analyze()
        assert result.total_predictions == 0

    def test_perfect_calibration(self, analyzer: CalibrationAnalyzer) -> None:
        # Record predictions where confidence matches accuracy
        for conf in [0.3, 0.5, 0.7, 0.9]:
            for i in range(100):
                was_correct = (i / 100.0) < conf
                analyzer.record(conf, was_correct, Action.BUY, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE)
        result = analyzer.analyze()
        assert result.total_predictions == 400
        assert result.expected_calibration_error < 0.15  # reasonable tolerance

    def test_overconfident_model(self, analyzer: CalibrationAnalyzer) -> None:
        # Always says 0.9 confidence but only 30% correct
        for _ in range(100):
            analyzer.record(0.9, False, Action.BUY, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE)
        for _ in range(30):
            analyzer.record(0.9, True, Action.BUY, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE)
        result = analyzer.analyze()
        assert result.overconfidence_ratio > 0

    def test_filter_by_model(self, analyzer: CalibrationAnalyzer) -> None:
        # Record for two models
        for _ in range(50):
            analyzer.record(0.8, True, Action.BUY, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE)
            analyzer.record(0.8, False, Action.BUY, ModelProvider.GEMINI, AgentArchitecture.SINGLE)

        ds_result = analyzer.analyze(model=ModelProvider.DEEPSEEK)
        ge_result = analyzer.analyze(model=ModelProvider.GEMINI)
        assert ds_result.total_predictions == 50
        assert ge_result.total_predictions == 50

    def test_analyze_all(self, analyzer: CalibrationAnalyzer) -> None:
        for _ in range(20):
            analyzer.record(0.7, True, Action.BUY, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE)
            analyzer.record(0.7, True, Action.BUY, ModelProvider.GEMINI, AgentArchitecture.MULTI)
        results = analyzer.analyze_all()
        assert len(results) >= 2

    def test_brier_score(self, analyzer: CalibrationAnalyzer) -> None:
        # Perfect predictions -> low Brier score
        for _ in range(50):
            analyzer.record(1.0, True, Action.BUY, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE)
        result = analyzer.analyze()
        assert result.brier_score < 0.01

    def test_clear(self, analyzer: CalibrationAnalyzer) -> None:
        analyzer.record(0.8, True, Action.BUY, ModelProvider.DEEPSEEK, AgentArchitecture.SINGLE)
        assert analyzer.record_count == 1
        analyzer.clear()
        assert analyzer.record_count == 0

    def test_well_calibrated_property(self) -> None:
        result = CalibrationResult(
            model=ModelProvider.DEEPSEEK,
            architecture=AgentArchitecture.SINGLE,
            bins=[],
            expected_calibration_error=0.03,
            max_calibration_error=0.05,
            overconfidence_ratio=0.2,
            total_predictions=100,
            brier_score=0.15,
        )
        assert result.is_well_calibrated  # ECE < 0.05
