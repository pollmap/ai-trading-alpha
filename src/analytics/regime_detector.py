"""Market regime detection -- classify bull/bear/sideways/high_vol/crash.

Detection rules (applied in priority order):

1. **Crash**: drawdown from rolling peak > 15 %
2. **High-volatility**: 20-day realised vol > 90th percentile of its own history
3. **Bull**: 20-day cumulative return > +5 %
4. **Bear**: 20-day cumulative return < -5 %
5. **Sideways**: everything else
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.core.logging import get_logger

log = get_logger(__name__)


# ── Regime Enum ──────────────────────────────────────────────────


class MarketRegime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_vol"
    CRASH = "crash"


# ── Detector ─────────────────────────────────────────────────────


class RegimeDetector:
    """Detect market regime from a price series.

    Detection methods (combined in priority order):
    1. Drawdown threshold for crash detection (> 15 % from peak).
    2. Volatility percentile for high-vol regime (> 90th percentile).
    3. SMA-crossover / rolling-return for trend classification.
    """

    def __init__(
        self,
        return_window: int = 20,
        bull_threshold: float = 0.05,
        bear_threshold: float = -0.05,
        vol_percentile: float = 90.0,
        crash_drawdown: float = 0.15,
    ) -> None:
        self._return_window: int = return_window
        self._bull_threshold: float = bull_threshold
        self._bear_threshold: float = bear_threshold
        self._vol_percentile: float = vol_percentile
        self._crash_drawdown: float = crash_drawdown

    # ── Public API ───────────────────────────────────────────────

    def detect(self, prices: pd.Series, vix: float | None = None) -> MarketRegime:
        """Classify the current market regime from a price series.

        Parameters
        ----------
        prices:
            Historical prices, chronologically sorted (oldest first).
            Must have at least ``return_window + 1`` data points.
        vix:
            Optional external VIX reading.  If provided and > 30, the
            detector uses it as supporting evidence for HIGH_VOL, but the
            primary signal remains the internal volatility calculation.

        Returns
        -------
        MarketRegime
        """
        if len(prices) < self._return_window + 1:
            log.warning(
                "insufficient_data_for_regime",
                length=len(prices),
                required=self._return_window + 1,
            )
            return MarketRegime.SIDEWAYS

        # 1. Crash detection -- drawdown from rolling peak
        peak = float(prices.cummax().iloc[-1])
        current = float(prices.iloc[-1])
        if peak > 0:
            drawdown = (peak - current) / peak
            if drawdown >= self._crash_drawdown:
                log.info("regime_crash", drawdown=round(drawdown, 4))
                return MarketRegime.CRASH

        # 2. High-volatility detection
        returns = prices.pct_change().dropna()
        if len(returns) >= self._return_window:
            rolling_vol = returns.rolling(window=self._return_window).std()
            current_vol = float(rolling_vol.iloc[-1])
            vol_threshold = float(np.nanpercentile(rolling_vol.dropna().values, self._vol_percentile))
            if current_vol > vol_threshold:
                log.info(
                    "regime_high_vol",
                    current_vol=round(current_vol, 6),
                    threshold=round(vol_threshold, 6),
                )
                return MarketRegime.HIGH_VOL

        # Optional VIX override for HIGH_VOL
        if vix is not None and vix > 30.0:
            log.info("regime_high_vol_vix", vix=vix)
            return MarketRegime.HIGH_VOL

        # 3. Trend via rolling return
        window_return = (current / float(prices.iloc[-self._return_window - 1])) - 1.0

        if window_return > self._bull_threshold:
            log.debug("regime_bull", window_return=round(window_return, 4))
            return MarketRegime.BULL
        if window_return < self._bear_threshold:
            log.debug("regime_bear", window_return=round(window_return, 4))
            return MarketRegime.BEAR

        log.debug("regime_sideways", window_return=round(window_return, 4))
        return MarketRegime.SIDEWAYS

    def detect_from_returns(self, returns: pd.Series) -> MarketRegime:
        """Classify regime directly from a return series.

        Reconstructs a synthetic price series (starting at 1.0) and delegates
        to :meth:`detect`.
        """
        if returns.empty:
            return MarketRegime.SIDEWAYS

        prices = (1.0 + returns).cumprod()
        prices = pd.concat([pd.Series([1.0]), prices]).reset_index(drop=True)
        return self.detect(prices)

    def get_regime_history(
        self,
        prices: pd.Series,
        window: int = 20,
    ) -> list[tuple[int, MarketRegime]]:
        """Compute a regime label for every position in *prices*.

        For each index ``i >= window``, the regime is determined by the
        sub-series ``prices[0:i+1]``.  Indices ``0`` through ``window - 1``
        are labelled ``SIDEWAYS`` (insufficient data).

        Returns
        -------
        list[tuple[int, MarketRegime]]
            Pairs of ``(index, regime)`` covering every row in *prices*.
        """
        history: list[tuple[int, MarketRegime]] = []

        # Not enough data for any regime detection
        for i in range(min(window, len(prices))):
            history.append((i, MarketRegime.SIDEWAYS))

        for i in range(window, len(prices)):
            sub_prices = prices.iloc[: i + 1]
            regime = self.detect(sub_prices)
            history.append((i, regime))

        log.debug("regime_history_computed", length=len(history))
        return history

    def to_prompt_context(self, regime: MarketRegime) -> str:
        """Generate an LLM-friendly regime context string."""
        descriptions: dict[MarketRegime, str] = {
            MarketRegime.BULL: (
                "Market regime: BULL. Prices trending upward with sustained positive "
                "momentum. Consider trend-following entries with tighter stop losses."
            ),
            MarketRegime.BEAR: (
                "Market regime: BEAR. Prices trending downward with sustained negative "
                "momentum. Exercise caution with new long positions; favour defensive "
                "and cash preservation."
            ),
            MarketRegime.SIDEWAYS: (
                "Market regime: SIDEWAYS. No clear directional trend. Range-bound "
                "trading expected. Mean-reversion strategies may outperform."
            ),
            MarketRegime.HIGH_VOL: (
                "Market regime: HIGH VOLATILITY. Elevated price swings detected. "
                "Reduce position sizes, widen stop losses, and avoid over-leveraging."
            ),
            MarketRegime.CRASH: (
                "Market regime: CRASH. Severe drawdown from recent peak (>15%). "
                "Capital preservation is paramount. Minimise exposure and maintain "
                "high cash reserves."
            ),
        }
        return descriptions[regime]


# ── Per-Regime Performance ───────────────────────────────────────


@dataclass
class RegimePerformance:
    """Performance statistics for a single market regime."""

    regime: MarketRegime
    n_periods: int
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


# ── Analyzer ─────────────────────────────────────────────────────


class RegimeAnalyzer:
    """Break down portfolio performance by market regime."""

    def analyze_by_regime(
        self,
        portfolio_values: list[float],
        regime_history: list[tuple[int, MarketRegime]],
    ) -> dict[MarketRegime, RegimePerformance]:
        """Segment *portfolio_values* by regime and compute per-regime stats.

        Parameters
        ----------
        portfolio_values:
            Time-ordered portfolio NAVs aligned 1:1 with *regime_history*.
        regime_history:
            Output of :meth:`RegimeDetector.get_regime_history`.

        Returns
        -------
        dict[MarketRegime, RegimePerformance]
        """
        if len(portfolio_values) != len(regime_history):
            log.warning(
                "length_mismatch",
                values=len(portfolio_values),
                regimes=len(regime_history),
            )
            # Truncate to the shorter length
            min_len = min(len(portfolio_values), len(regime_history))
            portfolio_values = portfolio_values[:min_len]
            regime_history = regime_history[:min_len]

        if not portfolio_values:
            return {}

        # Group indices by regime
        regime_indices: dict[MarketRegime, list[int]] = {}
        for idx, (_, regime) in enumerate(regime_history):
            regime_indices.setdefault(regime, []).append(idx)

        # Calculate per-period returns
        values_arr = np.array(portfolio_values, dtype=np.float64)
        period_returns = np.diff(values_arr) / values_arr[:-1]
        # period_returns[i] = return from index i to i+1

        results: dict[MarketRegime, RegimePerformance] = {}

        for regime, indices in regime_indices.items():
            # Collect returns for periods *starting* in this regime
            # (skip the last overall index since it has no forward return)
            regime_returns: list[float] = []
            for i in indices:
                if i < len(period_returns):
                    regime_returns.append(float(period_returns[i]))

            n_periods = len(regime_returns)
            if n_periods == 0:
                results[regime] = RegimePerformance(
                    regime=regime,
                    n_periods=0,
                    cumulative_return=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    win_rate=0.0,
                )
                continue

            returns_arr = np.array(regime_returns, dtype=np.float64)

            # Cumulative return
            cumulative_return = float(np.prod(1.0 + returns_arr) - 1.0)

            # Sharpe ratio (annualised, assuming 252 trading days)
            mean_ret = float(np.mean(returns_arr))
            std_ret = float(np.std(returns_arr, ddof=1)) if n_periods > 1 else 0.0
            sharpe_ratio = (
                (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0.0 else 0.0
            )

            # Max drawdown within this regime's equity curve
            equity = np.cumprod(1.0 + returns_arr)
            running_max = np.maximum.accumulate(equity)
            drawdowns = (running_max - equity) / running_max
            max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

            # Win rate
            wins = int(np.sum(returns_arr > 0.0))
            win_rate = wins / n_periods

            results[regime] = RegimePerformance(
                regime=regime,
                n_periods=n_periods,
                cumulative_return=round(cumulative_return, 6),
                sharpe_ratio=round(float(sharpe_ratio), 4),
                max_drawdown=round(max_drawdown, 6),
                win_rate=round(win_rate, 4),
            )

            log.debug(
                "regime_performance",
                regime=regime.value,
                n_periods=n_periods,
                cumulative_return=round(cumulative_return, 4),
                sharpe=round(float(sharpe_ratio), 2),
            )

        return results
