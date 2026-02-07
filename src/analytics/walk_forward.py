"""Walk-Forward Analysis and Monte Carlo simulation for strategy validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid_extensions import uuid7

import numpy as np

from src.core.logging import get_logger

log = get_logger(__name__)


# ── Default Thresholds ───────────────────────────────────────────
_DEFAULT_WINDOW_SIZE: int = 30
_DEFAULT_OOS_SIZE: int = 10
_DEFAULT_N_SIMULATIONS: int = 10_000
_DEFAULT_CONFIDENCE_LEVEL: float = 0.95
_ROBUSTNESS_EFFICIENCY_THRESHOLD: float = 0.5
_SIGNIFICANCE_ALPHA: float = 0.05
_MIN_WINDOW_DATA_POINTS: int = 5


@dataclass
class WalkForwardWindow:
    """Result of a single in-sample / out-of-sample window pair."""

    window_id: int = 0
    in_sample_start: int = 0
    in_sample_end: int = 0
    out_of_sample_start: int = 0
    out_of_sample_end: int = 0
    in_sample_return: float = 0.0
    out_of_sample_return: float = 0.0
    efficiency: float = 0.0  # OOS / IS (bounded, handles sign correctly)


@dataclass
class WalkForwardResult:
    """Aggregated result of walk-forward analysis."""

    result_id: str = field(default_factory=lambda: str(uuid7()))
    windows: list[WalkForwardWindow] = field(default_factory=list)
    avg_efficiency: float = 0.0
    oos_cumulative_return: float = 0.0
    is_robust: bool = False  # efficiency > 0.5
    total_data_points: int = 0
    n_windows: int = 0


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo simulation for statistical significance."""

    result_id: str = field(default_factory=lambda: str(uuid7()))
    actual_return: float = 0.0
    simulated_mean: float = 0.0
    simulated_std: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    percentile_rank: float = 0.0
    p_value: float = 0.0
    is_significant: bool = False
    n_simulations: int = 0


class WalkForwardAnalyzer:
    """Rolling window validation to detect overfitting.

    Splits portfolio value series into rolling in-sample (IS) and
    out-of-sample (OOS) windows. For each window, compares IS return
    to OOS return. If OOS performance tracks IS reasonably well
    (efficiency > 0.5), the strategy is considered robust.

    Walk-forward procedure:
    1. Start at index 0, take `window_size` values as in-sample
    2. Take next `oos_size` values as out-of-sample
    3. Calculate return for each segment
    4. Slide forward by `oos_size` and repeat
    5. Compute efficiency = OOS_return / IS_return for each window
    """

    def run(
        self,
        portfolio_values: list[float],
        window_size: int = _DEFAULT_WINDOW_SIZE,
        oos_size: int = _DEFAULT_OOS_SIZE,
    ) -> WalkForwardResult:
        """Run walk-forward analysis on portfolio value series.

        Args:
            portfolio_values: Time-ordered series of portfolio values.
            window_size: Number of data points in each in-sample window.
            oos_size: Number of data points in each out-of-sample window.

        Returns:
            WalkForwardResult with per-window details and aggregate metrics.
        """
        n: int = len(portfolio_values)
        min_required: int = window_size + oos_size

        if n < min_required:
            log.warning(
                "walk_forward_insufficient_data",
                data_points=n,
                minimum_required=min_required,
            )
            return WalkForwardResult(
                total_data_points=n,
                n_windows=0,
            )

        values: np.ndarray = np.array(portfolio_values, dtype=np.float64)
        windows: list[WalkForwardWindow] = []
        window_id: int = 0
        start: int = 0

        while start + window_size + oos_size <= n:
            is_start: int = start
            is_end: int = start + window_size
            oos_start: int = is_end
            oos_end: int = is_end + oos_size

            # In-sample return
            is_slice: np.ndarray = values[is_start:is_end]
            is_return: float = self._calculate_return(is_slice)

            # Out-of-sample return
            oos_slice: np.ndarray = values[oos_start:oos_end]
            oos_return: float = self._calculate_return(oos_slice)

            # Efficiency: how well OOS tracks IS
            efficiency: float = self._calculate_efficiency(is_return, oos_return)

            window: WalkForwardWindow = WalkForwardWindow(
                window_id=window_id,
                in_sample_start=is_start,
                in_sample_end=is_end - 1,
                out_of_sample_start=oos_start,
                out_of_sample_end=oos_end - 1,
                in_sample_return=round(is_return, 6),
                out_of_sample_return=round(oos_return, 6),
                efficiency=round(efficiency, 4),
            )
            windows.append(window)

            log.debug(
                "walk_forward_window",
                window_id=window_id,
                is_return=f"{is_return:.4f}",
                oos_return=f"{oos_return:.4f}",
                efficiency=f"{efficiency:.4f}",
            )

            window_id += 1
            start += oos_size  # slide forward by OOS size

        # ── Aggregate ─────────────────────────────────────────────
        efficiencies: list[float] = [w.efficiency for w in windows]
        avg_efficiency: float = (
            float(np.mean(efficiencies)) if efficiencies else 0.0
        )

        # Cumulative OOS return: chain all OOS windows
        oos_cumulative: float = self._chain_oos_returns(windows)

        is_robust: bool = avg_efficiency > _ROBUSTNESS_EFFICIENCY_THRESHOLD

        result: WalkForwardResult = WalkForwardResult(
            windows=windows,
            avg_efficiency=round(avg_efficiency, 4),
            oos_cumulative_return=round(oos_cumulative, 6),
            is_robust=is_robust,
            total_data_points=n,
            n_windows=len(windows),
        )

        log.info(
            "walk_forward_complete",
            n_windows=len(windows),
            avg_efficiency=f"{avg_efficiency:.4f}",
            oos_cumulative=f"{oos_cumulative:.4f}",
            is_robust=is_robust,
        )

        return result

    def _calculate_return(self, values: np.ndarray) -> float:
        """Calculate simple return over a value series.

        Args:
            values: Array of portfolio values.

        Returns:
            Return as a decimal (e.g. 0.05 for 5%).
        """
        if len(values) < 2 or values[0] == 0.0:
            return 0.0
        return float((values[-1] / values[0]) - 1.0)

    def _calculate_efficiency(
        self, is_return: float, oos_return: float
    ) -> float:
        """Calculate walk-forward efficiency (OOS / IS).

        Handles edge cases:
        - If IS return is zero, efficiency is 1.0 if OOS is also zero, else 0.0
        - If signs differ (IS positive but OOS negative), efficiency is negative
        - Clamps to [-2.0, 2.0] to avoid extreme outliers

        Args:
            is_return: In-sample return.
            oos_return: Out-of-sample return.

        Returns:
            Efficiency ratio, clamped to [-2.0, 2.0].
        """
        if abs(is_return) < 1e-10:
            return 1.0 if abs(oos_return) < 1e-10 else 0.0
        raw: float = oos_return / is_return
        return float(np.clip(raw, -2.0, 2.0))

    def _chain_oos_returns(self, windows: list[WalkForwardWindow]) -> float:
        """Chain out-of-sample returns across all windows.

        Compounds the OOS returns to get a cumulative OOS performance figure:
        cumulative = product of (1 + r_i) - 1

        Args:
            windows: List of walk-forward windows.

        Returns:
            Cumulative chained OOS return.
        """
        if not windows:
            return 0.0
        cumulative: float = 1.0
        for w in windows:
            cumulative *= (1.0 + w.out_of_sample_return)
        return cumulative - 1.0


class MonteCarloSimulator:
    """Monte Carlo simulation for statistical significance testing.

    Procedure:
    1. Take the actual daily return series
    2. Compute actual cumulative return
    3. Bootstrap-resample the returns N times (sample with replacement)
    4. For each bootstrap sample, compute cumulative return
    5. Build null distribution from all bootstrap cumulative returns
    6. Find where the actual return falls in this distribution
    7. If actual return is in the extreme tail, it is significant

    Bootstrap resampling is used instead of pure permutation because
    the product of (1 + r_i) is commutative -- every permutation yields
    the same cumulative return. Resampling with replacement creates
    genuinely different return paths by varying which returns appear
    and how many times each is drawn.
    """

    def simulate(
        self,
        returns: list[float],
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        confidence_level: float = _DEFAULT_CONFIDENCE_LEVEL,
    ) -> MonteCarloResult:
        """Bootstrap-resample returns N times to build null distribution,
        then check where actual performance falls.

        Args:
            returns: Daily return series (e.g. [0.01, -0.005, 0.02, ...]).
            n_simulations: Number of bootstrap resamples to generate.
            confidence_level: Confidence level for the interval (e.g. 0.95).

        Returns:
            MonteCarloResult with percentile rank, p-value, and significance flag.
        """
        if len(returns) < _MIN_WINDOW_DATA_POINTS:
            log.warning(
                "monte_carlo_insufficient_data",
                data_points=len(returns),
                minimum_required=_MIN_WINDOW_DATA_POINTS,
            )
            return MonteCarloResult(
                n_simulations=0,
            )

        returns_arr: np.ndarray = np.array(returns, dtype=np.float64)
        n_returns: int = len(returns_arr)

        # ── Actual cumulative return ──────────────────────────────
        actual_return: float = float(np.prod(1.0 + returns_arr) - 1.0)

        # ── Generate null distribution via bootstrap resampling ───
        # Sample with replacement: each simulation draws n_returns
        # values from the original return pool, allowing repeats.
        rng: np.random.Generator = np.random.default_rng()
        simulated_returns: np.ndarray = np.empty(n_simulations, dtype=np.float64)

        for i in range(n_simulations):
            bootstrap_sample: np.ndarray = rng.choice(
                returns_arr, size=n_returns, replace=True,
            )
            simulated_returns[i] = float(np.prod(1.0 + bootstrap_sample) - 1.0)

        # ── Statistics of null distribution ───────────────────────
        simulated_mean: float = float(np.mean(simulated_returns))
        simulated_std: float = float(
            np.std(simulated_returns, ddof=1) if n_simulations > 1 else 0.0
        )

        # Confidence interval on the null distribution
        alpha: float = 1.0 - confidence_level
        lower_pct: float = alpha / 2.0 * 100.0
        upper_pct: float = (1.0 - alpha / 2.0) * 100.0
        ci_lower: float = float(np.percentile(simulated_returns, lower_pct))
        ci_upper: float = float(np.percentile(simulated_returns, upper_pct))

        # ── Percentile rank of actual return ──────────────────────
        # What fraction of simulated returns are <= the actual return
        percentile_rank: float = float(
            np.mean(simulated_returns <= actual_return)
        )

        # ── P-value (one-tailed based on direction) ───────────────
        # Tests whether the actual return is significantly extreme
        if actual_return >= simulated_mean:
            # Right tail: how many simulations produced >= actual
            p_value: float = float(
                np.mean(simulated_returns >= actual_return)
            )
        else:
            # Left tail: how many simulations produced <= actual
            p_value = float(
                np.mean(simulated_returns <= actual_return)
            )

        is_significant: bool = p_value < _SIGNIFICANCE_ALPHA

        result: MonteCarloResult = MonteCarloResult(
            actual_return=round(actual_return, 6),
            simulated_mean=round(simulated_mean, 6),
            simulated_std=round(simulated_std, 6),
            confidence_interval=(round(ci_lower, 6), round(ci_upper, 6)),
            percentile_rank=round(percentile_rank, 4),
            p_value=round(p_value, 4),
            is_significant=is_significant,
            n_simulations=n_simulations,
        )

        log.info(
            "monte_carlo_complete",
            actual_return=f"{actual_return:.4f}",
            simulated_mean=f"{simulated_mean:.4f}",
            simulated_std=f"{simulated_std:.4f}",
            percentile_rank=f"{percentile_rank:.2%}",
            p_value=f"{p_value:.4f}",
            is_significant=is_significant,
            n_simulations=n_simulations,
        )

        return result
