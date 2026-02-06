"""Statistical comparison between models and architectures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between two models."""

    model_a: str
    model_b: str
    metric: str
    mean_a: float
    mean_b: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool   # p < 0.05
    test_used: str       # "welch_t" or "mann_whitney"


class ModelComparator:
    """Statistical comparison of model performance."""

    def pairwise_test(
        self,
        returns_a: list[float],
        returns_b: list[float],
        model_a: str = "A",
        model_b: str = "B",
        metric: str = "return",
        alpha: float = 0.05,
    ) -> ComparisonResult:
        """Compare two models using appropriate statistical test.

        Uses Shapiro-Wilk to test normality, then:
        - Normal: Welch's t-test
        - Non-normal: Mann-Whitney U test
        """
        a = np.array(returns_a, dtype=np.float64)
        b = np.array(returns_b, dtype=np.float64)

        if len(a) < 3 or len(b) < 3:
            return ComparisonResult(
                model_a=model_a, model_b=model_b, metric=metric,
                mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
                p_value=1.0, effect_size=0.0, significant=False,
                test_used="insufficient_data",
            )

        # Normality test
        normal_a = len(a) >= 8 and stats.shapiro(a).pvalue > 0.05
        normal_b = len(b) >= 8 and stats.shapiro(b).pvalue > 0.05

        if normal_a and normal_b:
            stat, p_value = stats.ttest_ind(a, b, equal_var=False)
            test_name = "welch_t"
        else:
            stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
            test_name = "mann_whitney"

        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        effect_size = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

        return ComparisonResult(
            model_a=model_a,
            model_b=model_b,
            metric=metric,
            mean_a=round(float(np.mean(a)), 4),
            mean_b=round(float(np.mean(b)), 4),
            p_value=round(float(p_value), 4),
            effect_size=round(float(effect_size), 4),
            significant=p_value < alpha,
            test_used=test_name,
        )

    def multi_model_test(
        self,
        groups: dict[str, list[float]],
        metric: str = "return",
    ) -> dict[str, object]:
        """Compare 3+ models simultaneously.

        Uses ANOVA (normal) or Kruskal-Wallis (non-normal).
        """
        arrays = [np.array(v, dtype=np.float64) for v in groups.values()]
        names = list(groups.keys())

        if any(len(a) < 3 for a in arrays):
            return {"test": "insufficient_data", "p_value": 1.0, "significant": False}

        # Check normality of all groups
        all_normal = all(
            len(a) >= 8 and stats.shapiro(a).pvalue > 0.05 for a in arrays
        )

        if all_normal:
            stat, p_value = stats.f_oneway(*arrays)
            test_name = "anova"
        else:
            stat, p_value = stats.kruskal(*arrays)
            test_name = "kruskal_wallis"

        return {
            "test": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "significant": p_value < 0.05,
            "groups": {name: round(float(np.mean(arr)), 4) for name, arr in zip(names, arrays)},
        }

    def bootstrap_confidence_interval(
        self,
        returns: list[float],
        n_resamples: int = 1000,
        confidence: float = 0.95,
    ) -> tuple[float, float, float]:
        """Bootstrap confidence interval for mean return.

        Returns: (mean, lower_bound, upper_bound)
        """
        data = np.array(returns, dtype=np.float64)
        means = np.array([
            np.mean(np.random.choice(data, size=len(data), replace=True))
            for _ in range(n_resamples)
        ])

        alpha = (1 - confidence) / 2
        lower = float(np.percentile(means, alpha * 100))
        upper = float(np.percentile(means, (1 - alpha) * 100))

        return round(float(np.mean(data)), 4), round(lower, 4), round(upper, 4)

    def regime_analysis(
        self,
        returns: list[float],
        threshold: float = 0.005,
    ) -> dict[str, dict[str, float]]:
        """Classify market regimes and analyze per-regime performance.

        Regimes: up (>threshold), down (<-threshold), sideways.
        """
        data = np.array(returns, dtype=np.float64)

        regimes: dict[str, list[float]] = {"up": [], "down": [], "sideways": []}
        for r in data:
            if r > threshold:
                regimes["up"].append(float(r))
            elif r < -threshold:
                regimes["down"].append(float(r))
            else:
                regimes["sideways"].append(float(r))

        result: dict[str, dict[str, float]] = {}
        for regime, values in regimes.items():
            if values:
                result[regime] = {
                    "count": len(values),
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
                    "pct_of_total": round(len(values) / len(data), 4),
                }
            else:
                result[regime] = {"count": 0, "mean": 0.0, "std": 0.0, "pct_of_total": 0.0}

        return result
