"""Multi-asset portfolio optimizer --- cross-market allocation.

Implements:
1. Mean-Variance Optimization (Markowitz)
2. Risk Parity
3. Black-Litterman with LLM views
4. Minimum Variance
5. Maximum Sharpe Ratio
6. Equal Weight

All numerical methods use pure numpy (no scipy dependency).  When matrix
inversions encounter singular covariance matrices the optimizer falls back
to equal-weight allocation and logs a warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.core.logging import get_logger

log = get_logger(__name__)


# ── Enums & Data Structures ─────────────────────────────────────


class OptimizationMethod(str, Enum):
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class AssetInfo:
    """Information about an asset for optimization."""

    symbol: str
    market: str
    expected_return: float  # annualized
    current_weight: float   # current portfolio weight


@dataclass
class LLMView:
    """LLM-generated view for Black-Litterman.

    Attributes:
        symbol: Asset symbol this view pertains to.
        expected_return: LLM's expected annualized return.
        confidence: 0-1 confidence score.  Higher -> lower uncertainty in
                    the Black-Litterman formulation.
    """

    symbol: str
    expected_return: float
    confidence: float  # 0-1, maps to uncertainty


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""

    method: OptimizationMethod
    weights: dict[str, float]     # symbol -> target weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float


# ── Optimizer ────────────────────────────────────────────────────


class MultiAssetOptimizer:
    """Cross-market portfolio optimizer.

    Supports six allocation methods that can be selected per call.
    Weights are always clipped to ``[min_weight, max_weight]`` and
    renormalised to sum to 1.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.04,
        min_weight: float = 0.0,
        max_weight: float = 0.40,
        risk_aversion: float = 2.5,
    ) -> None:
        self._rf: float = risk_free_rate
        self._min_w: float = min_weight
        self._max_w: float = max_weight
        self._risk_aversion: float = risk_aversion

    # ── Public entry point ──────────────────────────────────────

    def optimize(
        self,
        assets: list[AssetInfo],
        returns_matrix: np.ndarray,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        llm_views: list[LLMView] | None = None,
    ) -> OptimizationResult:
        """Run portfolio optimization.

        Args:
            assets: List of assets under consideration.
            returns_matrix: shape ``(n_periods, n_assets)`` of periodic
                returns (e.g. daily log-returns).
            method: Optimisation strategy to apply.
            llm_views: Required when *method* is ``BLACK_LITTERMAN``.

        Returns:
            :class:`OptimizationResult` with target weights and metrics.
        """
        n_assets = len(assets)
        if n_assets == 0:
            log.warning("optimize_empty_assets")
            return OptimizationResult(
                method=method,
                weights={},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=0.0,
                diversification_ratio=0.0,
            )

        if returns_matrix.ndim != 2 or returns_matrix.shape[1] != n_assets:
            log.error(
                "returns_matrix_shape_mismatch",
                expected_cols=n_assets,
                got_shape=returns_matrix.shape,
            )
            return self._equal_weight_result(assets, returns_matrix, method)

        cov = self.compute_covariance_matrix(returns_matrix)
        mu = np.array([a.expected_return for a in assets], dtype=np.float64)

        # Dispatch to requested method
        if method == OptimizationMethod.EQUAL_WEIGHT:
            raw_weights = self._equal_weight(n_assets)
        elif method == OptimizationMethod.MIN_VARIANCE:
            raw_weights = self._min_variance(cov, n_assets)
        elif method == OptimizationMethod.MEAN_VARIANCE:
            raw_weights = self._mean_variance(mu, cov, n_assets)
        elif method == OptimizationMethod.MAX_SHARPE:
            raw_weights = self._max_sharpe(mu, cov, n_assets)
        elif method == OptimizationMethod.RISK_PARITY:
            raw_weights = self._risk_parity(cov, n_assets)
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            if llm_views is None or len(llm_views) == 0:
                log.warning("black_litterman_no_views", msg="No LLM views provided; falling back to equal weight")
                raw_weights = self._equal_weight(n_assets)
            else:
                raw_weights = self._black_litterman(mu, cov, llm_views, assets)
        else:
            log.warning("unknown_optimization_method", method=method)
            raw_weights = self._equal_weight(n_assets)

        weights = self._clip_weights(raw_weights)

        # Build result
        symbols = [a.symbol for a in assets]
        weight_dict: dict[str, float] = {sym: float(w) for sym, w in zip(symbols, weights, strict=True)}

        port_return = float(weights @ mu)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        sharpe = (port_return - self._rf) / port_vol if port_vol > 1e-12 else 0.0

        # Diversification ratio = weighted avg vol / portfolio vol
        asset_vols = np.sqrt(np.diag(cov))
        weighted_avg_vol = float(weights @ asset_vols)
        div_ratio = weighted_avg_vol / port_vol if port_vol > 1e-12 else 1.0

        result = OptimizationResult(
            method=method,
            weights=weight_dict,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio,
        )

        log.info(
            "portfolio_optimized",
            method=method.value,
            n_assets=n_assets,
            expected_return=round(port_return, 6),
            expected_volatility=round(port_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            diversification_ratio=round(div_ratio, 4),
        )

        return result

    # ── Matrix utilities ────────────────────────────────────────

    def compute_correlation_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Compute the sample correlation matrix from a returns matrix.

        Args:
            returns_matrix: shape ``(n_periods, n_assets)``.

        Returns:
            Correlation matrix of shape ``(n_assets, n_assets)``.
        """
        if returns_matrix.shape[0] < 2:
            n = returns_matrix.shape[1]
            return np.eye(n, dtype=np.float64)
        corr = np.corrcoef(returns_matrix, rowvar=False)
        # np.corrcoef can return a scalar for 1-asset case
        if corr.ndim == 0:
            return np.array([[1.0]], dtype=np.float64)
        return corr.astype(np.float64)

    def compute_covariance_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Compute the sample covariance matrix from a returns matrix.

        Args:
            returns_matrix: shape ``(n_periods, n_assets)``.

        Returns:
            Covariance matrix of shape ``(n_assets, n_assets)``.
        """
        n = returns_matrix.shape[1]
        if returns_matrix.shape[0] < 2:
            return np.eye(n, dtype=np.float64) * 0.01
        cov = np.cov(returns_matrix, rowvar=False)
        # np.cov returns a scalar for single-asset
        if cov.ndim == 0:
            return np.array([[float(cov)]], dtype=np.float64)
        return cov.astype(np.float64)

    # ── Optimisation methods ────────────────────────────────────

    def _mean_variance(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """Analytical mean-variance: w* = (1/lambda) * Sigma^-1 * mu, normalised.

        Falls back to equal weight on singular covariance.
        """
        cov_inv = self._safe_inv(cov)
        if cov_inv is None:
            log.warning("mean_variance_singular_cov", msg="Falling back to equal weight")
            return self._equal_weight(n)

        raw = (1.0 / self._risk_aversion) * (cov_inv @ mu)
        # If all weights are non-positive, fall back
        if raw.sum() <= 0:
            log.warning("mean_variance_non_positive_weights", msg="Falling back to equal weight")
            return self._equal_weight(n)
        # Normalise to sum to 1 (long-only: set negatives to 0 first)
        raw = np.maximum(raw, 0.0)
        total = raw.sum()
        if total < 1e-12:
            return self._equal_weight(n)
        return raw / total

    def _risk_parity(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Risk-parity via iterative inverse-volatility reweighting.

        Target: each asset contributes equally to total portfolio risk.
        Uses iterative refinement (up to 100 rounds) starting from
        inverse-volatility weights.
        """
        diag = np.diag(cov)
        if np.any(diag <= 0):
            log.warning("risk_parity_non_positive_variance", msg="Falling back to equal weight")
            return self._equal_weight(n)

        # Start from inverse-volatility weights
        inv_vol = 1.0 / np.sqrt(diag)
        weights = inv_vol / inv_vol.sum()

        max_iterations: int = 100
        tolerance: float = 1e-8

        for iteration in range(max_iterations):
            # Marginal risk contribution: sigma_i = (cov @ w)_i * w_i
            sigma_w = cov @ weights
            mrc = weights * sigma_w  # marginal risk contributions
            total_risk = weights @ sigma_w

            if total_risk < 1e-16:
                return self._equal_weight(n)

            # Target: equal risk contribution = total_risk / n
            target_rc = total_risk / n
            risk_contrib = mrc / total_risk  # fractional risk contributions

            # Adjust: increase weight for under-contributing, decrease for over
            adjustment = target_rc / (mrc + 1e-16)
            new_weights = weights * adjustment
            new_weights = np.maximum(new_weights, 1e-10)
            new_weights = new_weights / new_weights.sum()

            # Check convergence
            if np.max(np.abs(new_weights - weights)) < tolerance:
                weights = new_weights
                log.debug("risk_parity_converged", iterations=iteration + 1)
                break

            weights = new_weights
        else:
            log.debug("risk_parity_max_iterations", iterations=max_iterations)

        return weights

    def _black_litterman(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        views: list[LLMView],
        assets: list[AssetInfo],
    ) -> np.ndarray:
        """Black-Litterman model with LLM confidence as views.

        Prior: equilibrium returns from CAPM  pi = lambda * Sigma * w_mkt
        Views are absolute:  P @ mu_BL = Q  with uncertainty Omega.
        Posterior: mu_BL = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 *
                           [(tau*Sigma)^-1*pi + P'*Omega^-1*Q]

        Then run mean-variance on the posterior.
        """
        n = len(assets)
        symbols = [a.symbol for a in assets]
        symbol_idx: dict[str, int] = {s: i for i, s in enumerate(symbols)}

        # Filter views to assets we actually have
        valid_views = [v for v in views if v.symbol in symbol_idx]
        if not valid_views:
            log.warning("black_litterman_no_matching_views", msg="Falling back to equal weight")
            return self._equal_weight(n)

        # ── Implied equilibrium returns (prior) ──
        tau: float = 0.05  # scaling factor for uncertainty of the mean
        # Market-cap weights (use current_weight as proxy)
        w_mkt = np.array([a.current_weight for a in assets], dtype=np.float64)
        w_mkt_sum = w_mkt.sum()
        if w_mkt_sum < 1e-12:
            w_mkt = np.ones(n, dtype=np.float64) / n
        else:
            w_mkt = w_mkt / w_mkt_sum

        pi = self._risk_aversion * (cov @ w_mkt)  # implied returns

        # ── Pick matrix P and view vector Q ──
        k = len(valid_views)
        P = np.zeros((k, n), dtype=np.float64)
        Q = np.zeros(k, dtype=np.float64)
        omega_diag = np.zeros(k, dtype=np.float64)

        for row, view in enumerate(valid_views):
            col = symbol_idx[view.symbol]
            P[row, col] = 1.0
            Q[row] = view.expected_return
            # Confidence -> uncertainty: higher confidence = lower omega
            # omega_i = (1 - confidence) * (P_i @ tau*Sigma @ P_i')
            conf = max(0.01, min(0.99, view.confidence))
            view_var = float(P[row] @ (tau * cov) @ P[row])
            omega_diag[row] = (1.0 - conf) / conf * max(view_var, 1e-8)

        Omega = np.diag(omega_diag)

        # ── Posterior ──
        tau_cov = tau * cov
        tau_cov_inv = self._safe_inv(tau_cov)
        omega_inv = self._safe_inv(Omega)
        if tau_cov_inv is None or omega_inv is None:
            log.warning("black_litterman_singular_matrix", msg="Falling back to equal weight")
            return self._equal_weight(n)

        # M = (tau*Sigma)^-1 + P' Omega^-1 P
        M = tau_cov_inv + P.T @ omega_inv @ P
        M_inv = self._safe_inv(M)
        if M_inv is None:
            log.warning("black_litterman_posterior_singular", msg="Falling back to equal weight")
            return self._equal_weight(n)

        mu_bl = M_inv @ (tau_cov_inv @ pi + P.T @ omega_inv @ Q)

        # Use posterior returns with mean-variance
        return self._mean_variance(mu_bl, cov, n)

    def _min_variance(self, cov: np.ndarray, n: int) -> np.ndarray:
        """Minimum variance portfolio: w* = Sigma^-1 * 1 / (1' * Sigma^-1 * 1).

        Falls back to equal weight on singular covariance.
        """
        cov_inv = self._safe_inv(cov)
        if cov_inv is None:
            log.warning("min_variance_singular_cov", msg="Falling back to equal weight")
            return self._equal_weight(n)

        ones = np.ones(n, dtype=np.float64)
        raw = cov_inv @ ones
        denom = float(ones @ cov_inv @ ones)
        if abs(denom) < 1e-12:
            return self._equal_weight(n)

        weights = raw / denom

        # Project to long-only
        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        if total < 1e-12:
            return self._equal_weight(n)
        return weights / total

    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
        """Maximum Sharpe ratio portfolio.

        Analytical: w* = Sigma^-1 * (mu - rf) / (1' * Sigma^-1 * (mu - rf))

        Falls back to equal weight on singular covariance or when all
        excess returns are non-positive.
        """
        cov_inv = self._safe_inv(cov)
        if cov_inv is None:
            log.warning("max_sharpe_singular_cov", msg="Falling back to equal weight")
            return self._equal_weight(n)

        excess = mu - self._rf
        raw = cov_inv @ excess
        ones = np.ones(n, dtype=np.float64)
        denom = float(ones @ raw)

        if abs(denom) < 1e-12:
            log.warning("max_sharpe_zero_denom", msg="Falling back to equal weight")
            return self._equal_weight(n)

        weights = raw / denom

        # Project to long-only
        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        if total < 1e-12:
            return self._equal_weight(n)
        return weights / total

    def _equal_weight(self, n: int) -> np.ndarray:
        """Uniform 1/N allocation."""
        if n == 0:
            return np.array([], dtype=np.float64)
        return np.ones(n, dtype=np.float64) / n

    # ── Weight post-processing ──────────────────────────────────

    def _clip_weights(self, weights: np.ndarray) -> np.ndarray:
        """Clip weights to [min_weight, max_weight] and renormalise.

        After clipping, renormalise so weights sum to 1. Repeat until
        convergence (at most 50 iterations to avoid infinite loops).
        """
        if len(weights) == 0:
            return weights

        w = weights.copy()
        max_iterations: int = 50

        for _ in range(max_iterations):
            w = np.clip(w, self._min_w, self._max_w)
            total = w.sum()
            if total < 1e-12:
                n = len(w)
                return np.ones(n, dtype=np.float64) / n
            w = w / total

            # Check if already within bounds after normalisation
            if np.all(w >= self._min_w - 1e-10) and np.all(w <= self._max_w + 1e-10):
                break

        return w

    # ── Internal helpers ────────────────────────────────────────

    @staticmethod
    def _safe_inv(matrix: np.ndarray) -> np.ndarray | None:
        """Invert a matrix, returning None for singular matrices.

        Uses a small ridge regularisation before giving up.
        """
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            pass

        # Attempt ridge regularisation
        try:
            n = matrix.shape[0]
            ridge = matrix + np.eye(n, dtype=np.float64) * 1e-6
            return np.linalg.inv(ridge)
        except np.linalg.LinAlgError:
            log.warning("matrix_inversion_failed", shape=matrix.shape)
            return None

    def _equal_weight_result(
        self,
        assets: list[AssetInfo],
        returns_matrix: np.ndarray,
        method: OptimizationMethod,
    ) -> OptimizationResult:
        """Build an equal-weight result as a safe fallback."""
        n = len(assets)
        weights = self._equal_weight(n)
        mu = np.array([a.expected_return for a in assets], dtype=np.float64)
        cov = self.compute_covariance_matrix(returns_matrix)

        port_return = float(weights @ mu) if n > 0 else 0.0
        port_vol = float(np.sqrt(weights @ cov @ weights)) if n > 0 else 0.0
        sharpe = (port_return - self._rf) / port_vol if port_vol > 1e-12 else 0.0

        asset_vols = np.sqrt(np.diag(cov)) if n > 0 else np.array([])
        weighted_avg_vol = float(weights @ asset_vols) if n > 0 else 0.0
        div_ratio = weighted_avg_vol / port_vol if port_vol > 1e-12 else 1.0

        return OptimizationResult(
            method=method,
            weights={a.symbol: float(w) for a, w in zip(assets, weights, strict=True)},
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            diversification_ratio=div_ratio,
        )
