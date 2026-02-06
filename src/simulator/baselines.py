"""Naive baseline strategies for benchmark comparison.

Each baseline generates :class:`TradingSignal` objects that flow through the
same order engine and portfolio manager as LLM-driven agents, ensuring a fair
comparison on identical market data.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from uuid_extensions import uuid7

from src.core.logging import get_logger
from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    MarketSnapshot,
    ModelProvider,
    PortfolioState,
    TradingSignal,
)

log = get_logger(__name__)

# Default model/architecture placeholders for baselines
_BASELINE_MODEL: ModelProvider = ModelProvider.GPT
_BASELINE_ARCH: AgentArchitecture = AgentArchitecture.SINGLE


# ── Helpers ──────────────────────────────────────────────────────


def _make_signal(
    snapshot: MarketSnapshot,
    symbol: str,
    action: Action,
    weight: float,
    confidence: float,
    reasoning: str,
) -> TradingSignal:
    """Construct a TradingSignal with standard baseline metadata."""
    return TradingSignal(
        signal_id=str(uuid7()),
        snapshot_id=snapshot.snapshot_id,
        timestamp=datetime.now(timezone.utc),
        symbol=symbol,
        market=snapshot.market,
        action=action,
        weight=weight,
        confidence=confidence,
        reasoning=reasoning,
        model=_BASELINE_MODEL,
        architecture=_BASELINE_ARCH,
    )


# ── Abstract Base ────────────────────────────────────────────────


class BaselineStrategy(ABC):
    """Interface for all naive baseline strategies."""

    @abstractmethod
    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> list[TradingSignal]:
        """Return a list of trading signals for the current cycle."""
        ...


# ── Buy & Hold ───────────────────────────────────────────────────


class BuyAndHoldBaseline(BaselineStrategy):
    """Buy equal weight on the first cycle, then hold forever.

    After the initial purchase no further signals are emitted (all HOLD).
    """

    def __init__(self) -> None:
        self._initialized: bool = False

    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> list[TradingSignal]:
        symbols = list(snapshot.symbols.keys())
        if not symbols:
            return []

        if not self._initialized:
            self._initialized = True
            weight = 1.0 / len(symbols) if symbols else 0.0
            signals: list[TradingSignal] = []
            for sym in symbols:
                signals.append(
                    _make_signal(
                        snapshot=snapshot,
                        symbol=sym,
                        action=Action.BUY,
                        weight=weight,
                        confidence=1.0,
                        reasoning=(
                            f"Buy-and-hold baseline: initial equal-weight allocation "
                            f"({weight:.2%} per symbol, {len(symbols)} symbols)"
                        ),
                    ),
                )
            log.info(
                "buy_and_hold_init",
                n_symbols=len(symbols),
                weight_per_symbol=round(weight, 4),
            )
            return signals

        # After first cycle: hold everything
        return [
            _make_signal(
                snapshot=snapshot,
                symbol=sym,
                action=Action.HOLD,
                weight=0.0,
                confidence=1.0,
                reasoning="Buy-and-hold baseline: holding existing positions, no action",
            )
            for sym in symbols
        ]


# ── Equal Weight Rebalance ───────────────────────────────────────


class EqualWeightRebalanceBaseline(BaselineStrategy):
    """Rebalance to equal weight every cycle.

    Emits BUY for underweight symbols, SELL for overweight, HOLD for those
    within a 1 percentage-point tolerance band.
    """

    def __init__(self, tolerance: float = 0.01) -> None:
        self._tolerance: float = tolerance

    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> list[TradingSignal]:
        symbols = list(snapshot.symbols.keys())
        if not symbols:
            return []

        target_weight = 1.0 / len(symbols)
        total_value = portfolio.total_value
        signals: list[TradingSignal] = []

        for sym in symbols:
            # Current weight of this symbol in portfolio
            current_weight = 0.0
            if total_value > 0.0 and sym in portfolio.positions:
                pos = portfolio.positions[sym]
                current_weight = (pos.current_price * pos.quantity) / total_value

            diff = target_weight - current_weight

            if diff > self._tolerance:
                action = Action.BUY
                reasoning = (
                    f"Equal-weight rebalance: {sym} underweight "
                    f"(current={current_weight:.2%}, target={target_weight:.2%})"
                )
            elif diff < -self._tolerance:
                action = Action.SELL
                reasoning = (
                    f"Equal-weight rebalance: {sym} overweight "
                    f"(current={current_weight:.2%}, target={target_weight:.2%})"
                )
            else:
                action = Action.HOLD
                reasoning = (
                    f"Equal-weight rebalance: {sym} within tolerance "
                    f"(current={current_weight:.2%}, target={target_weight:.2%})"
                )

            signals.append(
                _make_signal(
                    snapshot=snapshot,
                    symbol=sym,
                    action=action,
                    weight=target_weight,
                    confidence=0.8,
                    reasoning=reasoning,
                ),
            )

        log.debug(
            "equal_weight_rebalance",
            n_symbols=len(symbols),
            target_weight=round(target_weight, 4),
        )
        return signals


# ── Momentum ─────────────────────────────────────────────────────


class MomentumBaseline(BaselineStrategy):
    """Buy top-N performers by recent return, sell bottom-N.

    Uses the intra-day return (close vs. open) from the current snapshot as a
    momentum proxy.  Symbols in the middle are held.
    """

    def __init__(self, top_n: int = 3) -> None:
        self._top_n: int = top_n

    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> list[TradingSignal]:
        symbols = list(snapshot.symbols.keys())
        if not symbols:
            return []

        # Calculate return for each symbol
        returns: dict[str, float] = {}
        for sym, data in snapshot.symbols.items():
            if data.open > 0:
                returns[sym] = (data.close - data.open) / data.open
            else:
                returns[sym] = 0.0

        sorted_symbols = sorted(returns.keys(), key=lambda s: returns[s], reverse=True)
        top_n = min(self._top_n, len(sorted_symbols))

        top_set = set(sorted_symbols[:top_n])
        bottom_set = set(sorted_symbols[-top_n:]) if len(sorted_symbols) > top_n else set()
        # Avoid overlap when total symbols <= 2 * top_n
        bottom_set -= top_set

        weight = 1.0 / top_n if top_n > 0 else 0.0
        signals: list[TradingSignal] = []

        for sym in symbols:
            ret_pct = returns[sym] * 100
            if sym in top_set:
                signals.append(
                    _make_signal(
                        snapshot=snapshot,
                        symbol=sym,
                        action=Action.BUY,
                        weight=weight,
                        confidence=0.7,
                        reasoning=(
                            f"Momentum baseline: {sym} is top-{top_n} performer "
                            f"({ret_pct:+.2f}% intra-day return)"
                        ),
                    ),
                )
            elif sym in bottom_set:
                signals.append(
                    _make_signal(
                        snapshot=snapshot,
                        symbol=sym,
                        action=Action.SELL,
                        weight=0.0,
                        confidence=0.7,
                        reasoning=(
                            f"Momentum baseline: {sym} is bottom-{top_n} performer "
                            f"({ret_pct:+.2f}% intra-day return)"
                        ),
                    ),
                )
            else:
                signals.append(
                    _make_signal(
                        snapshot=snapshot,
                        symbol=sym,
                        action=Action.HOLD,
                        weight=0.0,
                        confidence=0.5,
                        reasoning=(
                            f"Momentum baseline: {sym} is mid-tier performer "
                            f"({ret_pct:+.2f}% intra-day return), holding"
                        ),
                    ),
                )

        log.debug(
            "momentum_signals",
            top=sorted_symbols[:top_n],
            bottom=list(bottom_set),
        )
        return signals


# ── Mean Reversion ───────────────────────────────────────────────


class MeanReversionBaseline(BaselineStrategy):
    """Buy oversold symbols, sell overbought ones.

    Uses the intra-day return as a proxy: symbols that fell the most are
    assumed to revert upward, and vice-versa.  If the snapshot ``metadata``
    contains pre-calculated RSI values (keyed ``rsi_<symbol>``), those are
    used instead (RSI < 30 = oversold, RSI > 70 = overbought).
    """

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        return_threshold: float = 0.02,
    ) -> None:
        self._rsi_oversold: float = rsi_oversold
        self._rsi_overbought: float = rsi_overbought
        self._return_threshold: float = return_threshold

    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> list[TradingSignal]:
        symbols = list(snapshot.symbols.keys())
        if not symbols:
            return []

        n_symbols = len(symbols)
        weight = 1.0 / n_symbols if n_symbols > 0 else 0.0
        signals: list[TradingSignal] = []

        for sym in symbols:
            data = snapshot.symbols[sym]

            # Try RSI from snapshot metadata first
            rsi_key = f"rsi_{sym}"
            rsi_val: float | None = None
            if rsi_key in snapshot.metadata:
                raw = snapshot.metadata[rsi_key]
                if isinstance(raw, (int, float)):
                    rsi_val = float(raw)

            if rsi_val is not None:
                # RSI-based mean reversion
                if rsi_val < self._rsi_oversold:
                    signals.append(
                        _make_signal(
                            snapshot=snapshot,
                            symbol=sym,
                            action=Action.BUY,
                            weight=weight,
                            confidence=0.6,
                            reasoning=(
                                f"Mean-reversion baseline: {sym} oversold "
                                f"(RSI={rsi_val:.1f} < {self._rsi_oversold})"
                            ),
                        ),
                    )
                elif rsi_val > self._rsi_overbought:
                    signals.append(
                        _make_signal(
                            snapshot=snapshot,
                            symbol=sym,
                            action=Action.SELL,
                            weight=0.0,
                            confidence=0.6,
                            reasoning=(
                                f"Mean-reversion baseline: {sym} overbought "
                                f"(RSI={rsi_val:.1f} > {self._rsi_overbought})"
                            ),
                        ),
                    )
                else:
                    signals.append(
                        _make_signal(
                            snapshot=snapshot,
                            symbol=sym,
                            action=Action.HOLD,
                            weight=0.0,
                            confidence=0.5,
                            reasoning=(
                                f"Mean-reversion baseline: {sym} neutral "
                                f"(RSI={rsi_val:.1f}, range {self._rsi_oversold}-{self._rsi_overbought})"
                            ),
                        ),
                    )
            else:
                # Fallback: use intra-day return as proxy
                intra_return = (
                    (data.close - data.open) / data.open if data.open > 0 else 0.0
                )
                if intra_return < -self._return_threshold:
                    signals.append(
                        _make_signal(
                            snapshot=snapshot,
                            symbol=sym,
                            action=Action.BUY,
                            weight=weight,
                            confidence=0.5,
                            reasoning=(
                                f"Mean-reversion baseline: {sym} dropped "
                                f"{intra_return * 100:+.2f}% intra-day, expecting reversion"
                            ),
                        ),
                    )
                elif intra_return > self._return_threshold:
                    signals.append(
                        _make_signal(
                            snapshot=snapshot,
                            symbol=sym,
                            action=Action.SELL,
                            weight=0.0,
                            confidence=0.5,
                            reasoning=(
                                f"Mean-reversion baseline: {sym} surged "
                                f"{intra_return * 100:+.2f}% intra-day, expecting reversion"
                            ),
                        ),
                    )
                else:
                    signals.append(
                        _make_signal(
                            snapshot=snapshot,
                            symbol=sym,
                            action=Action.HOLD,
                            weight=0.0,
                            confidence=0.5,
                            reasoning=(
                                f"Mean-reversion baseline: {sym} within normal range "
                                f"({intra_return * 100:+.2f}% intra-day), holding"
                            ),
                        ),
                    )

        log.debug("mean_reversion_signals", n_signals=len(signals))
        return signals


# ── Random ───────────────────────────────────────────────────────


class RandomBaseline(BaselineStrategy):
    """Random BUY/SELL/HOLD with random weight.

    Serves as a statistical null hypothesis: any strategy should beat random.
    Uses a fixed seed for reproducibility when ``seed`` is provided.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng: random.Random = random.Random(seed)

    def generate_signals(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> list[TradingSignal]:
        symbols = list(snapshot.symbols.keys())
        if not symbols:
            return []

        actions = [Action.BUY, Action.SELL, Action.HOLD]
        signals: list[TradingSignal] = []

        for sym in symbols:
            action = self._rng.choice(actions)
            weight = round(self._rng.uniform(0.0, 1.0 / len(symbols)), 4)
            confidence = round(self._rng.uniform(0.0, 1.0), 2)

            signals.append(
                _make_signal(
                    snapshot=snapshot,
                    symbol=sym,
                    action=action,
                    weight=weight,
                    confidence=confidence,
                    reasoning=(
                        f"Random baseline: {sym} -> {action.value} "
                        f"(weight={weight:.4f}, confidence={confidence:.2f}), "
                        f"null-hypothesis control"
                    ),
                ),
            )

        log.debug("random_signals", n_signals=len(signals))
        return signals
