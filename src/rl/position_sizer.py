"""RL-based position sizer — adjusts signal weight using learned policy.

Uses a simplified Q-learning approach where:
- State: (volatility_regime, signal_confidence_bin, drawdown_bin, win_rate_bin)
- Action: scale_factor in {0.25, 0.50, 0.75, 1.00, 1.25}
- Reward: risk-adjusted PnL of the resulting trade
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.core.logging import get_logger
from src.core.types import TradingSignal, PortfolioState

log = get_logger(__name__)


# ── State Discretisation ────────────────────────────────────────

def _bin_confidence(confidence: float) -> str:
    if confidence < 0.3:
        return "low"
    if confidence < 0.6:
        return "mid"
    if confidence < 0.8:
        return "high"
    return "very_high"


def _bin_drawdown(drawdown_pct: float) -> str:
    if drawdown_pct < 0.03:
        return "none"
    if drawdown_pct < 0.08:
        return "mild"
    if drawdown_pct < 0.15:
        return "moderate"
    return "severe"


def _bin_win_rate(win_rate: float) -> str:
    if win_rate < 0.35:
        return "poor"
    if win_rate < 0.50:
        return "below_avg"
    if win_rate < 0.65:
        return "above_avg"
    return "strong"


@dataclass
class RLState:
    """Discretised state for Q-table lookup."""
    volatility_regime: str  # "low" | "normal" | "high" | "extreme"
    confidence_bin: str
    drawdown_bin: str
    win_rate_bin: str

    @property
    def key(self) -> str:
        return f"{self.volatility_regime}|{self.confidence_bin}|{self.drawdown_bin}|{self.win_rate_bin}"


# ── Scale Actions ──────────────────────────────────────────────

SCALE_ACTIONS: list[float] = [0.25, 0.50, 0.75, 1.00, 1.25]


@dataclass
class SizingDecision:
    """Output of the position sizer."""
    original_weight: float
    scaled_weight: float
    scale_factor: float
    state: RLState
    exploration: bool  # True if action was random (epsilon-greedy)


class RLPositionSizer:
    """Q-learning position sizer.

    Adjusts the weight of a TradingSignal based on the current market state.
    Can be trained online (during live benchmark) or offline from replay data.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ) -> None:
        self._lr: float = learning_rate
        self._gamma: float = discount_factor
        self._epsilon: float = epsilon
        self._epsilon_decay: float = epsilon_decay
        self._min_epsilon: float = min_epsilon
        self._q_table: dict[str, list[float]] = {}  # state_key -> Q-values per action
        self._rng: random.Random = random.Random(42)
        self._step_count: int = 0

    def _get_q_values(self, state_key: str) -> list[float]:
        if state_key not in self._q_table:
            self._q_table[state_key] = [0.0] * len(SCALE_ACTIONS)
        return self._q_table[state_key]

    def decide(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        volatility_regime: str = "normal",
        recent_win_rate: float = 0.50,
    ) -> SizingDecision:
        """Choose a position scale factor for the given signal.

        Args:
            signal: The trading signal whose weight to adjust.
            portfolio: Current portfolio state.
            volatility_regime: Current market volatility regime string.
            recent_win_rate: Win rate over recent lookback window.

        Returns:
            SizingDecision with original and adjusted weight.
        """
        drawdown = self._calculate_drawdown(portfolio)
        state = RLState(
            volatility_regime=volatility_regime,
            confidence_bin=_bin_confidence(signal.confidence),
            drawdown_bin=_bin_drawdown(drawdown),
            win_rate_bin=_bin_win_rate(recent_win_rate),
        )

        q_values = self._get_q_values(state.key)
        exploration = False

        # Epsilon-greedy action selection
        if self._rng.random() < self._epsilon:
            action_idx = self._rng.randint(0, len(SCALE_ACTIONS) - 1)
            exploration = True
        else:
            action_idx = int(np.argmax(q_values))

        scale_factor = SCALE_ACTIONS[action_idx]
        scaled_weight = min(1.0, signal.weight * scale_factor)

        decision = SizingDecision(
            original_weight=signal.weight,
            scaled_weight=scaled_weight,
            scale_factor=scale_factor,
            state=state,
            exploration=exploration,
        )

        log.debug(
            "rl_sizing_decision",
            state=state.key,
            scale_factor=scale_factor,
            original_weight=signal.weight,
            scaled_weight=scaled_weight,
            exploration=exploration,
            epsilon=self._epsilon,
        )

        return decision

    def update(
        self,
        state_key: str,
        action_idx: int,
        reward: float,
        next_state_key: str,
    ) -> None:
        """Q-learning update rule.

        Args:
            state_key: State key where action was taken.
            action_idx: Index into SCALE_ACTIONS.
            reward: Observed reward (risk-adjusted PnL).
            next_state_key: State key after the action.
        """
        q_values = self._get_q_values(state_key)
        next_q_values = self._get_q_values(next_state_key)

        # Q(s,a) <- Q(s,a) + lr * [r + gamma * max(Q(s',a')) - Q(s,a)]
        current_q = q_values[action_idx]
        max_next_q = max(next_q_values)
        new_q = current_q + self._lr * (reward + self._gamma * max_next_q - current_q)
        q_values[action_idx] = new_q

        # Decay epsilon
        self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)
        self._step_count += 1

        log.debug(
            "rl_q_update",
            state=state_key,
            action_idx=action_idx,
            reward=reward,
            old_q=current_q,
            new_q=new_q,
            epsilon=self._epsilon,
            step=self._step_count,
        )

    def save(self, path: Path) -> None:
        """Persist Q-table to disk."""
        data = {
            "q_table": self._q_table,
            "epsilon": self._epsilon,
            "step_count": self._step_count,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        log.info("rl_model_saved", path=str(path), states=len(self._q_table))

    def load(self, path: Path) -> None:
        """Load Q-table from disk."""
        if not path.exists():
            log.warning("rl_model_not_found", path=str(path))
            return
        data = json.loads(path.read_text())
        self._q_table = data.get("q_table", {})
        self._epsilon = data.get("epsilon", self._epsilon)
        self._step_count = data.get("step_count", 0)
        log.info("rl_model_loaded", path=str(path), states=len(self._q_table))

    @staticmethod
    def _calculate_drawdown(portfolio: PortfolioState) -> float:
        if portfolio.initial_capital <= 0:
            return 0.0
        total = portfolio.total_value
        if total >= portfolio.initial_capital:
            return 0.0
        return (portfolio.initial_capital - total) / portfolio.initial_capital

    @property
    def q_table_size(self) -> int:
        return len(self._q_table)

    @property
    def total_steps(self) -> int:
        return self._step_count
