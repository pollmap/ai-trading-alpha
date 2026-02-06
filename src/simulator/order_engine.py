"""Order execution engine — converts trading signals into virtual trades.

The engine is **stateless**: it evaluates a signal against the current
portfolio and market config, then returns a ``Trade`` record describing
what happened.  The caller (typically :class:`PortfolioManager`) is
responsible for applying the trade to portfolio state.

Fill-price model::

    BUY  fill_price = close_price * (1 + slippage_rate)
    SELL fill_price = close_price * (1 - slippage_rate)

Commission model (from ``market_config``)::

    BUY  commission = fill_price * quantity * commission.buy
    SELL commission = fill_price * quantity * (commission.sell + commission.tax)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from uuid_extensions import uuid7

from src.core.constants import (
    DEFAULT_MAX_POSITION_WEIGHT,
    DEFAULT_MIN_CASH_RATIO,
    DEFAULT_SLIPPAGE,
)
from src.core.exceptions import InsufficientFundsError, PositionLimitError
from src.core.logging import get_logger
from src.core.types import Action, PortfolioState, TradingSignal
from src.simulator.pnl_calculator import PnLCalculator

log = get_logger(__name__)


# ── Trade Record ────────────────────────────────────────────────


@dataclass(frozen=True)
class Trade:
    """Immutable record of a single executed virtual trade."""

    trade_id: str
    signal_id: str
    timestamp: datetime
    symbol: str
    action: Action
    quantity: float
    price: float       # fill price (after slippage)
    commission: float   # total commission + tax
    slippage: float     # total slippage cost (monetary)
    realized_pnl: float

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            msg = f"Trade quantity must be positive, got {self.quantity}"
            raise ValueError(msg)
        if self.price <= 0:
            msg = f"Trade price must be positive, got {self.price}"
            raise ValueError(msg)
        if self.commission < 0:
            msg = f"Trade commission must be non-negative, got {self.commission}"
            raise ValueError(msg)
        if self.slippage < 0:
            msg = f"Trade slippage must be non-negative, got {self.slippage}"
            raise ValueError(msg)
        if not self.trade_id:
            msg = "Trade trade_id must not be empty"
            raise ValueError(msg)
        if not self.signal_id:
            msg = "Trade signal_id must not be empty"
            raise ValueError(msg)


# ── Order Engine ────────────────────────────────────────────────


class OrderEngine:
    """Stateless order execution engine.

    Converts a :class:`TradingSignal` into a :class:`Trade` while
    enforcing risk limits (max position weight, min cash ratio).

    The engine does **not** mutate portfolio state.  It returns a
    ``Trade`` describing what should happen, and the caller applies
    it via :meth:`PortfolioManager.update_position`.

    Args:
        pnl_calculator: Optional ``PnLCalculator`` instance.
            Defaults to a fresh instance if not provided.
    """

    def __init__(self, pnl_calculator: PnLCalculator | None = None) -> None:
        self._pnl = pnl_calculator or PnLCalculator()

    # ── Public API ──────────────────────────────────────────────

    def execute_signal(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        market_config: dict[str, object],
        close_price: float,
    ) -> Trade | None:
        """Evaluate a trading signal and, if valid, create a ``Trade``.

        For **BUY** signals ``signal.weight`` is the *target portfolio
        weight* for the symbol (e.g. 0.20 = 20 % of total value).  The
        engine calculates how many additional units are needed and clamps
        to risk limits.

        For **SELL** signals ``signal.weight`` is the *target portfolio
        weight* after the reduction (0.0 = full liquidation).

        Args:
            signal: The signal to execute.
            portfolio: Current portfolio state (pre-trade).
            market_config: Market-specific config dict from
                ``config/markets.yaml`` (keys: ``commission``,
                ``slippage``, ``max_position_weight``,
                ``min_cash_ratio``).
            close_price: Reference close price for the symbol at
                signal time (from the ``MarketSnapshot``).

        Returns:
            A ``Trade`` if the signal resulted in an order, or ``None``
            for HOLD signals and signals that cannot be executed (e.g.
            already at target weight, zero portfolio value).

        Raises:
            InsufficientFundsError: If a BUY would produce negative
                cash even after clamping.
            PositionLimitError: If the post-trade position weight
                exceeds the configured maximum.
        """
        if signal.action == Action.HOLD:
            log.debug(
                "hold_signal_skipped",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
            )
            return None

        if close_price <= 0:
            log.warning(
                "invalid_close_price",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                close_price=close_price,
            )
            return None

        # ── Parse market config with safe defaults ──────────────
        slippage_rate = float(
            market_config.get("slippage", DEFAULT_SLIPPAGE),  # type: ignore[arg-type]
        )
        max_weight = float(
            market_config.get(  # type: ignore[arg-type]
                "max_position_weight", DEFAULT_MAX_POSITION_WEIGHT,
            ),
        )
        min_cash_ratio = float(
            market_config.get("min_cash_ratio", DEFAULT_MIN_CASH_RATIO),  # type: ignore[arg-type]
        )

        comm_cfg = market_config.get("commission", {})
        if not isinstance(comm_cfg, dict):
            comm_cfg = {}
        buy_comm_rate = float(comm_cfg.get("buy", 0.0))  # type: ignore[arg-type]
        sell_comm_rate = float(comm_cfg.get("sell", 0.0))  # type: ignore[arg-type]
        tax_rate = float(comm_cfg.get("tax", 0.0))  # type: ignore[arg-type]

        # ── Dispatch ────────────────────────────────────────────
        if signal.action == Action.BUY:
            return self._execute_buy(
                signal=signal,
                portfolio=portfolio,
                close_price=close_price,
                slippage_rate=slippage_rate,
                buy_comm_rate=buy_comm_rate,
                max_weight=max_weight,
                min_cash_ratio=min_cash_ratio,
            )

        return self._execute_sell(
            signal=signal,
            portfolio=portfolio,
            close_price=close_price,
            slippage_rate=slippage_rate,
            sell_comm_rate=sell_comm_rate,
            tax_rate=tax_rate,
        )

    # ── BUY Logic ───────────────────────────────────────────────

    def _execute_buy(
        self,
        *,
        signal: TradingSignal,
        portfolio: PortfolioState,
        close_price: float,
        slippage_rate: float,
        buy_comm_rate: float,
        max_weight: float,
        min_cash_ratio: float,
    ) -> Trade | None:
        """Build a BUY trade, clamping quantity to respect risk limits."""
        fill_price = close_price * (1.0 + slippage_rate)
        total_value = portfolio.total_value

        if total_value <= 0:
            log.warning(
                "buy_rejected_zero_portfolio",
                signal_id=signal.signal_id,
            )
            return None

        # Current position value at the fill price
        existing_pos = portfolio.positions.get(signal.symbol)
        existing_value = (
            existing_pos.quantity * fill_price if existing_pos else 0.0
        )

        # Target position value, clamped to max_position_weight
        effective_weight = min(signal.weight, max_weight)
        target_value = effective_weight * total_value
        additional_value = target_value - existing_value

        if additional_value <= 0:
            log.info(
                "buy_skipped_already_at_target",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                existing_weight=round(existing_value / total_value, 4),
                target_weight=effective_weight,
            )
            return None

        # Raw quantity before cash constraints
        raw_quantity = additional_value / fill_price

        # Max cash we can spend while preserving min_cash_ratio.
        # After trade: new_cash = cash - cost
        # Require:     new_cash >= min_cash_ratio * total_value
        # Therefore:   cost <= cash - min_cash_ratio * total_value
        max_spendable = portfolio.cash - min_cash_ratio * total_value
        if max_spendable <= 0:
            log.info(
                "buy_rejected_cash_ratio",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                cash=round(portfolio.cash, 2),
                min_cash_ratio=min_cash_ratio,
                total_value=round(total_value, 2),
            )
            return None

        # Total cost = fill_price * qty + commission
        #            = fill_price * qty * (1 + buy_comm_rate)
        # Clamp quantity so total cost <= max_spendable
        raw_cost = fill_price * raw_quantity * (1.0 + buy_comm_rate)
        if raw_cost > max_spendable:
            quantity = max_spendable / (fill_price * (1.0 + buy_comm_rate))
            log.info(
                "buy_quantity_clamped_by_cash_ratio",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                raw_quantity=round(raw_quantity, 6),
                clamped_quantity=round(quantity, 6),
            )
        else:
            quantity = raw_quantity

        if quantity <= 0:
            return None

        commission = fill_price * quantity * buy_comm_rate
        total_cost = fill_price * quantity + commission

        # Final cash validation
        if total_cost > portfolio.cash:
            raise InsufficientFundsError(
                f"BUY {signal.symbol}: need {total_cost:.2f}, "
                f"have {portfolio.cash:.2f}",
                context={
                    "signal_id": signal.signal_id,
                    "symbol": signal.symbol,
                    "total_cost": total_cost,
                    "cash": portfolio.cash,
                },
            )

        # Final position-weight validation (1 % tolerance for float rounding)
        new_pos_value = existing_value + fill_price * quantity
        post_trade_total = total_value - commission
        if post_trade_total > 0:
            post_weight = new_pos_value / post_trade_total
            if post_weight > max_weight * 1.01:
                raise PositionLimitError(
                    f"BUY {signal.symbol}: post-trade weight "
                    f"{post_weight:.2%} exceeds max {max_weight:.2%}",
                    context={
                        "signal_id": signal.signal_id,
                        "symbol": signal.symbol,
                        "post_weight": post_weight,
                        "max_weight": max_weight,
                    },
                )

        slippage_cost = slippage_rate * close_price * quantity

        trade = Trade(
            trade_id=str(uuid7()),
            signal_id=signal.signal_id,
            timestamp=datetime.now(timezone.utc),
            symbol=signal.symbol,
            action=Action.BUY,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage_cost,
            realized_pnl=0.0,
        )

        log.info(
            "buy_trade_executed",
            trade_id=trade.trade_id,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            quantity=round(quantity, 6),
            fill_price=round(fill_price, 6),
            commission=round(commission, 4),
            slippage=round(slippage_cost, 4),
            total_cost=round(total_cost, 4),
        )
        return trade

    # ── SELL Logic ──────────────────────────────────────────────

    def _execute_sell(
        self,
        *,
        signal: TradingSignal,
        portfolio: PortfolioState,
        close_price: float,
        slippage_rate: float,
        sell_comm_rate: float,
        tax_rate: float,
    ) -> Trade | None:
        """Build a SELL trade.

        ``signal.weight`` is interpreted as the *target portfolio weight*
        after the sell.  A weight of 0.0 means full liquidation.
        """
        existing_pos = portfolio.positions.get(signal.symbol)
        if existing_pos is None or existing_pos.quantity <= 0:
            log.warning(
                "sell_rejected_no_position",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
            )
            return None

        fill_price = close_price * (1.0 - slippage_rate)
        total_value = portfolio.total_value

        # Target position value after sell
        target_value = signal.weight * total_value if total_value > 0 else 0.0
        current_value = existing_pos.quantity * fill_price
        excess_value = current_value - target_value

        if excess_value <= 0:
            log.info(
                "sell_skipped_below_target",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                current_weight=round(
                    current_value / total_value if total_value > 0 else 0.0,
                    4,
                ),
                target_weight=signal.weight,
            )
            return None

        quantity = excess_value / fill_price
        # Clamp to available holdings
        quantity = min(quantity, existing_pos.quantity)

        if quantity <= 0:
            return None

        combined_sell_rate = sell_comm_rate + tax_rate
        commission = fill_price * quantity * combined_sell_rate

        # Realized PnL (net of commission)
        realized_pnl = self._pnl.calculate_realized_pnl(
            sell_price=fill_price,
            avg_entry_price=existing_pos.avg_entry_price,
            quantity=quantity,
            commission=commission,
        )

        slippage_cost = slippage_rate * close_price * quantity

        trade = Trade(
            trade_id=str(uuid7()),
            signal_id=signal.signal_id,
            timestamp=datetime.now(timezone.utc),
            symbol=signal.symbol,
            action=Action.SELL,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage_cost,
            realized_pnl=realized_pnl,
        )

        log.info(
            "sell_trade_executed",
            trade_id=trade.trade_id,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            quantity=round(quantity, 6),
            fill_price=round(fill_price, 6),
            commission=round(commission, 4),
            slippage=round(slippage_cost, 4),
            realized_pnl=round(realized_pnl, 4),
        )
        return trade
