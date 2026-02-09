"""DQN-based position sizer with GPU acceleration.

Uses Deep Q-Network with PyTorch for function approximation instead of a
tabular Q-table.  When PyTorch is unavailable the class transparently falls
back to the existing :class:`RLPositionSizer` (Q-table) implementation so
that callers never need to change their code.

Architecture
------------
- Input:  16-dim one-hot vector encoding the discretised RL state
          (4 volatility + 4 confidence + 4 drawdown + 4 win_rate)
- Hidden: two fully-connected layers (64, 32) with ReLU
- Output: Q-values for the 5 scale actions

Training uses standard DQN ingredients:
  * Experience-replay buffer  (capacity 10 000)
  * Target network with soft (Polyak) update  (tau = 0.005)
  * Mini-batch gradient descent  (batch 64)
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.logging import get_logger
from src.core.types import PortfolioState, TradingSignal
from src.rl.position_sizer import (
    SCALE_ACTIONS,
    RLPositionSizer,
    RLState,
    SizingDecision,
    _bin_confidence,
    _bin_drawdown,
    _bin_win_rate,
)

log = get_logger(__name__)

# ── Attempt to import PyTorch ───────────────────────────────────

_TORCH_AVAILABLE: bool = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]


# ── One-hot encoding helpers ────────────────────────────────────

_VOLATILITY_BINS: list[str] = ["low", "normal", "high", "extreme"]
_CONFIDENCE_BINS: list[str] = ["low", "mid", "high", "very_high"]
_DRAWDOWN_BINS: list[str] = ["none", "mild", "moderate", "severe"]
_WIN_RATE_BINS: list[str] = ["poor", "below_avg", "above_avg", "strong"]

STATE_DIM: int = len(_VOLATILITY_BINS) + len(_CONFIDENCE_BINS) + len(_DRAWDOWN_BINS) + len(_WIN_RATE_BINS)  # 16
ACTION_DIM: int = len(SCALE_ACTIONS)  # 5


def _encode_state(state: RLState) -> np.ndarray:
    """Convert an :class:`RLState` into a 16-dim one-hot vector."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    offset = 0

    for bins, value in [
        (_VOLATILITY_BINS, state.volatility_regime),
        (_CONFIDENCE_BINS, state.confidence_bin),
        (_DRAWDOWN_BINS, state.drawdown_bin),
        (_WIN_RATE_BINS, state.win_rate_bin),
    ]:
        idx = bins.index(value) if value in bins else 0
        vec[offset + idx] = 1.0
        offset += len(bins)

    return vec


def _encode_state_key(state_key: str) -> np.ndarray:
    """Encode a pipe-delimited state key string to a one-hot vector."""
    parts = state_key.split("|")
    if len(parts) != 4:
        return np.zeros(STATE_DIM, dtype=np.float32)
    state = RLState(
        volatility_regime=parts[0],
        confidence_bin=parts[1],
        drawdown_bin=parts[2],
        win_rate_bin=parts[3],
    )
    return _encode_state(state)


# ── Experience replay ───────────────────────────────────────────

@dataclass
class Transition:
    """Single experience tuple for replay."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray


class ReplayBuffer:
    """Fixed-capacity circular buffer for experience replay."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._rng: random.Random = random.Random(42)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self._rng.sample(list(self._buffer), min(batch_size, len(self._buffer)))

    def __len__(self) -> int:
        return len(self._buffer)


# ── DQN network ─────────────────────────────────────────────────

def _build_dqn(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, int],
) -> nn.Module:
    """Build a simple 2-hidden-layer DQN."""
    h1, h2 = hidden_sizes
    model = nn.Sequential(
        nn.Linear(input_dim, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, output_dim),
    )
    return model


# ── GPU Position Sizer ──────────────────────────────────────────

class GPUPositionSizer:
    """DQN-based position sizer with GPU acceleration.

    Drop-in replacement for :class:`RLPositionSizer` with better
    generalisation through neural-network function approximation.

    When PyTorch is not installed the class transparently delegates all
    calls to an internal :class:`RLPositionSizer` instance so that the
    rest of the system never notices.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        batch_size: int = 64,
        buffer_capacity: int = 10_000,
        target_update_tau: float = 0.005,
        hidden_sizes: tuple[int, int] = (64, 32),
    ) -> None:
        self._lr: float = learning_rate
        self._gamma: float = discount_factor
        self._epsilon: float = epsilon
        self._epsilon_decay: float = epsilon_decay
        self._min_epsilon: float = min_epsilon
        self._batch_size: int = batch_size
        self._tau: float = target_update_tau
        self._hidden_sizes: tuple[int, int] = hidden_sizes
        self._step_count: int = 0
        self._rng: random.Random = random.Random(42)

        # Fallback flag
        self._use_fallback: bool = not _TORCH_AVAILABLE

        if self._use_fallback:
            log.warning(
                "pytorch_not_available",
                msg="PyTorch not installed — falling back to Q-table RLPositionSizer",
            )
            self._fallback = RLPositionSizer(
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                min_epsilon=min_epsilon,
            )
            return

        # ── PyTorch-based DQN setup ─────────────────────────────
        self._device: torch.device = self._detect_device()
        log.info("gpu_position_sizer_init", device=str(self._device))

        self._policy_net: nn.Module = _build_dqn(STATE_DIM, ACTION_DIM, hidden_sizes).to(self._device)
        self._target_net: nn.Module = _build_dqn(STATE_DIM, ACTION_DIM, hidden_sizes).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer: optim.Adam = optim.Adam(self._policy_net.parameters(), lr=self._lr)
        self._replay: ReplayBuffer = ReplayBuffer(capacity=buffer_capacity)

    # ── Device detection ────────────────────────────────────────

    @staticmethod
    def _detect_device() -> torch.device:
        """Auto-detect the best available device: CUDA > MPS > CPU."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            log.info("device_detected", device="cuda", gpu=torch.cuda.get_device_name(0))
            return device
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            log.info("device_detected", device="mps")
            return device
        log.info("device_detected", device="cpu")
        return torch.device("cpu")

    # ── Public interface (matches RLPositionSizer) ──────────────

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
        if self._use_fallback:
            return self._fallback.decide(signal, portfolio, volatility_regime, recent_win_rate)

        drawdown = self._calculate_drawdown(portfolio)
        state = RLState(
            volatility_regime=volatility_regime,
            confidence_bin=_bin_confidence(signal.confidence),
            drawdown_bin=_bin_drawdown(drawdown),
            win_rate_bin=_bin_win_rate(recent_win_rate),
        )

        state_vec = _encode_state(state)
        exploration = False

        # Epsilon-greedy action selection
        if self._rng.random() < self._epsilon:
            action_idx = self._rng.randint(0, ACTION_DIM - 1)
            exploration = True
        else:
            action_idx = self._predict_action(state_vec)

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
            "dqn_sizing_decision",
            state=state.key,
            scale_factor=scale_factor,
            original_weight=signal.weight,
            scaled_weight=scaled_weight,
            exploration=exploration,
            epsilon=self._epsilon,
            device=str(self._device),
        )

        return decision

    def update(
        self,
        state_key: str,
        action_idx: int,
        reward: float,
        next_state_key: str,
    ) -> None:
        """Record a transition and train the DQN.

        Args:
            state_key: State key where action was taken.
            action_idx: Index into SCALE_ACTIONS.
            reward: Observed reward (risk-adjusted PnL).
            next_state_key: State key after the action.
        """
        if self._use_fallback:
            self._fallback.update(state_key, action_idx, reward, next_state_key)
            return

        state_vec = _encode_state_key(state_key)
        next_state_vec = _encode_state_key(next_state_key)

        # Store transition in replay buffer
        self._replay.push(Transition(
            state=state_vec,
            action=action_idx,
            reward=reward,
            next_state=next_state_vec,
        ))

        # Train if we have enough samples
        if len(self._replay) >= self._batch_size:
            self._train_step()

        # Soft-update target network
        self._soft_update_target()

        # Decay epsilon
        self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)
        self._step_count += 1

        log.debug(
            "dqn_update",
            state=state_key,
            action_idx=action_idx,
            reward=reward,
            buffer_size=len(self._replay),
            epsilon=self._epsilon,
            step=self._step_count,
        )

    # ── Save / Load ─────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist model weights (or Q-table in fallback mode) to disk."""
        if self._use_fallback:
            self._fallback.save(path)
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, object] = {
            "policy_state_dict": {k: v.cpu().tolist() for k, v in self._policy_net.state_dict().items()},
            "target_state_dict": {k: v.cpu().tolist() for k, v in self._target_net.state_dict().items()},
            "epsilon": self._epsilon,
            "step_count": self._step_count,
            "hidden_sizes": list(self._hidden_sizes),
            "learning_rate": self._lr,
            "discount_factor": self._gamma,
        }
        path.write_text(json.dumps(checkpoint, indent=2))
        log.info("dqn_model_saved", path=str(path), step=self._step_count)

    def load(self, path: Path) -> None:
        """Load model weights (or Q-table in fallback mode) from disk."""
        if self._use_fallback:
            self._fallback.load(path)
            return

        if not path.exists():
            log.warning("dqn_model_not_found", path=str(path))
            return

        data = json.loads(path.read_text())

        # Reconstruct state dicts from lists -> tensors
        policy_sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in data["policy_state_dict"].items()}
        target_sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in data["target_state_dict"].items()}

        self._policy_net.load_state_dict(policy_sd)
        self._target_net.load_state_dict(target_sd)
        self._policy_net.to(self._device)
        self._target_net.to(self._device)
        self._target_net.eval()

        self._epsilon = data.get("epsilon", self._epsilon)
        self._step_count = data.get("step_count", 0)

        log.info("dqn_model_loaded", path=str(path), step=self._step_count)

    # ── Properties ──────────────────────────────────────────────

    @property
    def total_steps(self) -> int:
        """Total number of update steps performed."""
        if self._use_fallback:
            return self._fallback.total_steps
        return self._step_count

    @property
    def buffer_size(self) -> int:
        """Current replay buffer size (0 in fallback mode)."""
        if self._use_fallback:
            return 0
        return len(self._replay)

    @property
    def device_name(self) -> str:
        """String name of the compute device."""
        if self._use_fallback:
            return "cpu_fallback"
        return str(self._device)

    @property
    def using_fallback(self) -> bool:
        """Whether the instance is using Q-table fallback."""
        return self._use_fallback

    # ── Private methods ─────────────────────────────────────────

    def _predict_action(self, state_vec: np.ndarray) -> int:
        """Run a forward pass to get the greedy action index."""
        with torch.no_grad():
            state_t = torch.tensor(state_vec, dtype=torch.float32, device=self._device).unsqueeze(0)
            q_values = self._policy_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def _train_step(self) -> None:
        """Sample a mini-batch from replay and perform one gradient step."""
        batch = self._replay.sample(self._batch_size)

        states = torch.tensor(
            np.array([t.state for t in batch]),
            dtype=torch.float32,
            device=self._device,
        )
        actions = torch.tensor(
            [t.action for t in batch],
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(1)
        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self._device,
        )
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self._device,
        )

        # Q(s, a) from policy network
        q_values = self._policy_net(states).gather(1, actions).squeeze(1)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self._target_net(next_states).max(dim=1).values

        # Bellman target: r + gamma * max Q(s', a')
        target = rewards + self._gamma * next_q_values

        # Huber loss (smooth L1)
        loss = nn.functional.smooth_l1_loss(q_values, target)

        self._optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self._policy_net.parameters(), max_norm=1.0)
        self._optimizer.step()

    def _soft_update_target(self) -> None:
        """Polyak averaging: target <- tau * policy + (1-tau) * target."""
        for target_param, policy_param in zip(
            self._target_net.parameters(),
            self._policy_net.parameters(),
            strict=True,
        ):
            target_param.data.copy_(
                self._tau * policy_param.data + (1.0 - self._tau) * target_param.data
            )

    @staticmethod
    def _calculate_drawdown(portfolio: PortfolioState) -> float:
        """Calculate current drawdown from initial capital."""
        if portfolio.initial_capital <= 0:
            return 0.0
        total = portfolio.total_value
        if total >= portfolio.initial_capital:
            return 0.0
        return (portfolio.initial_capital - total) / portfolio.initial_capital
