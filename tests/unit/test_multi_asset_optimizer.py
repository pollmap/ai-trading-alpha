"""Tests for multi-asset portfolio optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from src.simulator.multi_asset_optimizer import (
    MultiAssetOptimizer,
    OptimizationMethod,
    AssetInfo,
    LLMView,
    OptimizationResult,
)


def _make_assets(n: int = 4) -> list[AssetInfo]:
    names = ["AAPL", "TSM", "SHEL.L", "MC.PA", "GC=F", "TLT"][:n]
    markets = ["US", "JPX", "LSE", "EURONEXT", "COMMODITIES", "BOND"][:n]
    returns = [0.12, 0.10, 0.08, 0.15, 0.06, 0.04][:n]
    return [
        AssetInfo(symbol=s, market=m, expected_return=r, current_weight=1.0 / n)
        for s, m, r in zip(names, markets, returns)
    ]


def _make_returns(n_periods: int = 252, n_assets: int = 4, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # Correlated returns
    base = rng.randn(n_periods, n_assets) * 0.015  # ~1.5% daily vol
    # Add some positive drift
    drift = np.array([0.0005, 0.0004, 0.0003, 0.0006])[:n_assets]
    return base + drift


class TestMultiAssetOptimizer:
    """Tests for MultiAssetOptimizer."""

    def test_equal_weight(self) -> None:
        opt = MultiAssetOptimizer()
        assets = _make_assets(4)
        returns = _make_returns(252, 4)

        result = opt.optimize(assets, returns, method=OptimizationMethod.EQUAL_WEIGHT)

        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == 4
        for w in result.weights.values():
            assert abs(w - 0.25) < 0.001
        assert abs(sum(result.weights.values()) - 1.0) < 0.001

    def test_min_variance(self) -> None:
        opt = MultiAssetOptimizer()
        assets = _make_assets(4)
        returns = _make_returns(252, 4)

        result = opt.optimize(assets, returns, method=OptimizationMethod.MIN_VARIANCE)

        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        # All weights should be non-negative after clipping
        for w in result.weights.values():
            assert w >= -0.01  # small tolerance for float

    def test_max_sharpe(self) -> None:
        opt = MultiAssetOptimizer()
        assets = _make_assets(4)
        returns = _make_returns(252, 4)

        result = opt.optimize(assets, returns, method=OptimizationMethod.MAX_SHARPE)

        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.sharpe_ratio != 0.0

    def test_risk_parity(self) -> None:
        opt = MultiAssetOptimizer()
        assets = _make_assets(4)
        returns = _make_returns(252, 4)

        result = opt.optimize(assets, returns, method=OptimizationMethod.RISK_PARITY)

        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        # Risk parity should give roughly similar weights for similar-vol assets
        weights = list(result.weights.values())
        assert max(weights) < 0.5  # no single asset dominates

    def test_mean_variance(self) -> None:
        opt = MultiAssetOptimizer()
        assets = _make_assets(4)
        returns = _make_returns(252, 4)

        result = opt.optimize(assets, returns, method=OptimizationMethod.MEAN_VARIANCE)

        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_black_litterman(self) -> None:
        opt = MultiAssetOptimizer()
        assets = _make_assets(4)
        returns = _make_returns(252, 4)
        views = [
            LLMView(symbol="AAPL", expected_return=0.15, confidence=0.8),
            LLMView(symbol="SHEL.L", expected_return=0.03, confidence=0.6),
        ]

        result = opt.optimize(
            assets, returns,
            method=OptimizationMethod.BLACK_LITTERMAN,
            llm_views=views,
        )

        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_weight_clipping(self) -> None:
        opt = MultiAssetOptimizer(min_weight=0.05, max_weight=0.40)
        assets = _make_assets(4)
        returns = _make_returns(252, 4)

        result = opt.optimize(assets, returns, method=OptimizationMethod.MAX_SHARPE)

        for w in result.weights.values():
            assert w >= -0.01  # approximately >= min_weight (after normalization)
            assert w <= 0.45   # approximately <= max_weight (after normalization)

    def test_correlation_matrix(self) -> None:
        opt = MultiAssetOptimizer()
        returns = _make_returns(252, 4)
        corr = opt.compute_correlation_matrix(returns)

        assert corr.shape == (4, 4)
        # Diagonal should be 1.0
        for i in range(4):
            assert abs(corr[i, i] - 1.0) < 0.001
        # Should be symmetric
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_covariance_matrix(self) -> None:
        opt = MultiAssetOptimizer()
        returns = _make_returns(252, 4)
        cov = opt.compute_covariance_matrix(returns)

        assert cov.shape == (4, 4)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        # Diagonal should be positive (variances)
        for i in range(4):
            assert cov[i, i] > 0

    def test_single_asset(self) -> None:
        opt = MultiAssetOptimizer()
        assets = _make_assets(1)
        returns = _make_returns(252, 1)

        result = opt.optimize(assets, returns, method=OptimizationMethod.EQUAL_WEIGHT)
        assert abs(list(result.weights.values())[0] - 1.0) < 0.001

    def test_result_fields(self) -> None:
        opt = MultiAssetOptimizer()
        assets = _make_assets(4)
        returns = _make_returns(252, 4)

        result = opt.optimize(assets, returns, method=OptimizationMethod.MAX_SHARPE)

        assert result.method == OptimizationMethod.MAX_SHARPE
        assert isinstance(result.expected_return, float)
        assert isinstance(result.expected_volatility, float)
        assert result.expected_volatility > 0
        assert isinstance(result.diversification_ratio, float)
