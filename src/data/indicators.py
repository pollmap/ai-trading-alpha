"""Technical indicators engine -- pre-calculate indicators for LLM prompts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging import get_logger

log = get_logger(__name__)


# ── Dataclass ────────────────────────────────────────────────────


@dataclass
class TechnicalIndicators:
    """Computed technical indicators for a single symbol."""

    symbol: str
    rsi_14: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    ema_12: float | None = None
    ema_26: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    volume_sma_20: float | None = None
    volume_ratio: float | None = None
    atr_14: float | None = None
    historical_volatility_20: float | None = None

    def to_prompt_line(self) -> str:
        """Compact one-line summary suitable for LLM prompts."""
        parts: list[str] = [self.symbol + ":"]
        if self.rsi_14 is not None:
            parts.append(f"RSI={self.rsi_14:.1f}")
        if self.macd_histogram is not None:
            parts.append(f"MACD_H={self.macd_histogram:+.2f}")
        if self.bb_width is not None:
            parts.append(f"BB_W={self.bb_width:.3f}")
        if self.sma_50 is not None and self.sma_200 is not None:
            arrow = "\u2191" if self.sma_50 > self.sma_200 else "\u2193"
            parts.append(f"SMA50/200={arrow}")
        if self.volume_ratio is not None:
            parts.append(f"Vol={self.volume_ratio:.1f}x")
        if self.atr_14 is not None:
            parts.append(f"ATR={self.atr_14:.2f}")
        return " ".join(parts)


# ── Engine ───────────────────────────────────────────────────────


class IndicatorEngine:
    """Calculate technical indicators from OHLCV DataFrames.

    Expected DataFrame columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
    Rows must be sorted in chronological order (oldest first).
    """

    def calculate(self, symbol: str, ohlcv: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all indicators for *symbol*.

        Returns :class:`TechnicalIndicators` with ``None`` for any indicator
        whose minimum data requirement is not met.
        """
        if ohlcv.empty:
            log.warning("empty_ohlcv", symbol=symbol)
            return TechnicalIndicators(symbol=symbol)

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(ohlcv.columns)
        if missing:
            log.warning("missing_ohlcv_columns", symbol=symbol, missing=list(missing))
            return TechnicalIndicators(symbol=symbol)

        closes: pd.Series = ohlcv["close"]
        highs: pd.Series = ohlcv["high"]
        lows: pd.Series = ohlcv["low"]
        volumes: pd.Series = ohlcv["volume"]

        rsi_14 = self._rsi(closes, period=14)
        sma_20 = self._sma(closes, period=20)
        sma_50 = self._sma(closes, period=50)
        sma_200 = self._sma(closes, period=200)
        ema_12 = self._ema(closes, period=12)
        ema_26 = self._ema(closes, period=26)
        macd_line, macd_signal, macd_histogram = self._macd(closes)
        bb_upper, bb_middle, bb_lower, bb_width = self._bollinger(closes)
        atr_14 = self._atr(highs, lows, closes, period=14)
        volume_sma_20 = self._sma(volumes, period=20)

        # Volume ratio: current volume / 20-day SMA of volume
        volume_ratio: float | None = None
        if volume_sma_20 is not None and volume_sma_20 > 0:
            current_volume = float(volumes.iloc[-1])
            volume_ratio = current_volume / volume_sma_20

        # Historical volatility: annualised std-dev of 20-day log returns
        historical_volatility_20 = self._historical_volatility(closes, period=20)

        result = TechnicalIndicators(
            symbol=symbol,
            rsi_14=rsi_14,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_width=bb_width,
            volume_sma_20=volume_sma_20,
            volume_ratio=volume_ratio,
            atr_14=atr_14,
            historical_volatility_20=historical_volatility_20,
        )
        log.debug("indicators_calculated", symbol=symbol, rsi=rsi_14, atr=atr_14)
        return result

    def calculate_batch(
        self, ohlcv_by_symbol: dict[str, pd.DataFrame],
    ) -> dict[str, TechnicalIndicators]:
        """Calculate indicators for multiple symbols at once."""
        results: dict[str, TechnicalIndicators] = {}
        for symbol, ohlcv in ohlcv_by_symbol.items():
            results[symbol] = self.calculate(symbol, ohlcv)
        log.info("batch_indicators_calculated", count=len(results))
        return results

    # ── Private helpers ──────────────────────────────────────────

    @staticmethod
    def _rsi(closes: pd.Series, period: int = 14) -> float | None:
        """Relative Strength Index using Wilder smoothing."""
        if len(closes) < period + 1:
            return None

        deltas: pd.Series = closes.diff()
        gains: pd.Series = deltas.clip(lower=0.0)
        losses: pd.Series = (-deltas).clip(lower=0.0)

        # Wilder smoothing: first value is SMA, rest use exponential decay
        avg_gain = float(gains.iloc[1 : period + 1].mean())
        avg_loss = float(losses.iloc[1 : period + 1].mean())

        for i in range(period + 1, len(closes)):
            avg_gain = (avg_gain * (period - 1) + float(gains.iloc[i])) / period
            avg_loss = (avg_loss * (period - 1) + float(losses.iloc[i])) / period

        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _sma(series: pd.Series, period: int) -> float | None:
        """Simple Moving Average -- latest value."""
        if len(series) < period:
            return None
        return float(series.iloc[-period:].mean())

    @staticmethod
    def _ema(series: pd.Series, period: int) -> float | None:
        """Exponential Moving Average -- latest value."""
        if len(series) < period:
            return None
        ema_series: pd.Series = series.ewm(span=period, adjust=False).mean()
        return float(ema_series.iloc[-1])

    @staticmethod
    def _macd(
        closes: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> tuple[float | None, float | None, float | None]:
        """MACD line, signal line, and histogram.

        Requires at least *slow + signal_period* data points.
        """
        min_required = slow + signal_period
        if len(closes) < min_required:
            return None, None, None

        ema_fast: pd.Series = closes.ewm(span=fast, adjust=False).mean()
        ema_slow: pd.Series = closes.ewm(span=slow, adjust=False).mean()
        macd_line_series: pd.Series = ema_fast - ema_slow
        signal_line_series: pd.Series = macd_line_series.ewm(
            span=signal_period, adjust=False,
        ).mean()
        histogram_series: pd.Series = macd_line_series - signal_line_series

        return (
            float(macd_line_series.iloc[-1]),
            float(signal_line_series.iloc[-1]),
            float(histogram_series.iloc[-1]),
        )

    @staticmethod
    def _bollinger(
        closes: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Bollinger Bands: upper, middle, lower, width.

        Width is defined as ``(upper - lower) / middle``.
        """
        if len(closes) < period:
            return None, None, None, None

        window: pd.Series = closes.iloc[-period:]
        middle = float(window.mean())
        std = float(window.std(ddof=0))

        upper = middle + std_dev * std
        lower = middle - std_dev * std
        width = (upper - lower) / middle if middle != 0.0 else None

        return upper, middle, lower, width

    @staticmethod
    def _atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> float | None:
        """Average True Range using Wilder smoothing."""
        if len(close) < period + 1:
            return None

        # True Range components
        prev_close: pd.Series = close.shift(1)
        tr1: pd.Series = high - low
        tr2: pd.Series = (high - prev_close).abs()
        tr3: pd.Series = (low - prev_close).abs()
        true_range: pd.DataFrame = pd.concat([tr1, tr2, tr3], axis=1)
        tr: pd.Series = true_range.max(axis=1)

        # Wilder smoothing: first ATR is SMA of first *period* TRs (skip index 0)
        atr_value = float(tr.iloc[1 : period + 1].mean())
        for i in range(period + 1, len(tr)):
            atr_value = (atr_value * (period - 1) + float(tr.iloc[i])) / period

        return atr_value

    @staticmethod
    def _historical_volatility(
        closes: pd.Series, period: int = 20, annualisation_factor: int = 252,
    ) -> float | None:
        """Annualised historical volatility from log returns."""
        if len(closes) < period + 1:
            return None

        log_returns: np.ndarray = np.log(
            closes.iloc[-period - 1 :].values[1:] / closes.iloc[-period - 1 :].values[:-1],
        )
        daily_std = float(np.std(log_returns, ddof=1))
        return daily_std * np.sqrt(annualisation_factor)
