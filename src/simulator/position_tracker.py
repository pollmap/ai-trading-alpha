"""Position lifecycle tracking — open, accumulate, reduce, close.

Maintains an immutable audit trail of every position change for
post-hoc analysis and regulatory-style record keeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from uuid_extensions import uuid7

from src.core.logging import get_logger
from src.core.types import Position

log = get_logger(__name__)


# ── Position Change Record ──────────────────────────────────────


@dataclass(frozen=True)
class PositionChangeRecord:
    """Immutable record of a single position change event."""

    record_id: str
    timestamp: datetime
    symbol: str
    action: str  # "OPEN" | "ADD" | "REDUCE" | "CLOSE"
    quantity_delta: float
    price: float
    avg_entry_before: float
    avg_entry_after: float
    remaining_quantity: float
    realized_pnl: float

    def __post_init__(self) -> None:
        if self.action not in {"OPEN", "ADD", "REDUCE", "CLOSE"}:
            msg = f"Invalid position change action: {self.action}"
            raise ValueError(msg)
        if not self.symbol:
            msg = "PositionChangeRecord symbol must not be empty"
            raise ValueError(msg)


# ── Position Tracker ────────────────────────────────────────────


class PositionTracker:
    """Tracks position lifecycle changes and maintains an audit trail.

    Each instance tracks changes for a single portfolio.  The tracker
    is purely in-memory; persistence is the caller's responsibility.
    """

    def __init__(self) -> None:
        self._history: list[PositionChangeRecord] = []

    # ── Public API ──────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
    ) -> Position:
        """Create a brand-new position for *symbol*.

        Args:
            symbol: Ticker / asset identifier.
            quantity: Number of units to buy (must be > 0).
            price: Execution price per unit (must be > 0).

        Returns:
            A fresh ``Position`` with *quantity* units at *price*.

        Raises:
            ValueError: If *quantity* or *price* is non-positive.
        """
        if quantity <= 0:
            msg = f"open_position quantity must be positive, got {quantity}"
            raise ValueError(msg)
        if price <= 0:
            msg = f"open_position price must be positive, got {price}"
            raise ValueError(msg)

        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_entry_price=price,
            current_price=price,
        )

        self._record(
            symbol=symbol,
            action="OPEN",
            quantity_delta=quantity,
            price=price,
            avg_entry_before=0.0,
            avg_entry_after=price,
            remaining_quantity=quantity,
            realized_pnl=0.0,
        )

        log.info(
            "position_opened",
            symbol=symbol,
            quantity=quantity,
            price=price,
        )
        return position

    def add_to_position(
        self,
        position: Position,
        quantity: float,
        price: float,
    ) -> Position:
        """Add to an existing position, recalculating the weighted-average entry.

        Args:
            position: The current ``Position`` to augment.
            quantity: Additional units to buy (must be > 0).
            price: Execution price per unit (must be > 0).

        Returns:
            Updated ``Position`` with the new average entry price.

        Raises:
            ValueError: If *quantity* or *price* is non-positive.
        """
        if quantity <= 0:
            msg = f"add_to_position quantity must be positive, got {quantity}"
            raise ValueError(msg)
        if price <= 0:
            msg = f"add_to_position price must be positive, got {price}"
            raise ValueError(msg)

        old_avg = position.avg_entry_price
        old_qty = position.quantity
        new_qty = old_qty + quantity

        # Weighted-average entry price
        new_avg = ((old_avg * old_qty) + (price * quantity)) / new_qty

        updated = Position(
            symbol=position.symbol,
            quantity=new_qty,
            avg_entry_price=new_avg,
            current_price=price,
            realized_pnl=position.realized_pnl,
        )

        self._record(
            symbol=position.symbol,
            action="ADD",
            quantity_delta=quantity,
            price=price,
            avg_entry_before=old_avg,
            avg_entry_after=new_avg,
            remaining_quantity=new_qty,
            realized_pnl=0.0,
        )

        log.info(
            "position_added",
            symbol=position.symbol,
            added_qty=quantity,
            price=price,
            new_avg_entry=round(new_avg, 6),
            total_quantity=new_qty,
        )
        return updated

    def reduce_position(
        self,
        position: Position,
        quantity: float,
        sell_price: float,
    ) -> tuple[Position, float]:
        """Reduce (or close) an existing position.

        Args:
            position: The current ``Position`` to reduce.
            quantity: Number of units to sell (must be > 0).
            sell_price: Execution price per unit (must be > 0).

        Returns:
            A ``(updated_position, realized_pnl)`` tuple.  The realized
            PnL is the *gross* profit before commission.

        Raises:
            ValueError: If *quantity* or *sell_price* is non-positive, or
                if *quantity* exceeds the held amount.
        """
        if quantity <= 0:
            msg = f"reduce_position quantity must be positive, got {quantity}"
            raise ValueError(msg)
        if sell_price <= 0:
            msg = f"reduce_position sell_price must be positive, got {sell_price}"
            raise ValueError(msg)
        if quantity > position.quantity:
            msg = (
                f"Cannot reduce {position.symbol} by {quantity}; "
                f"only {position.quantity} held"
            )
            raise ValueError(msg)

        realized_pnl = (sell_price - position.avg_entry_price) * quantity
        remaining_qty = position.quantity - quantity
        action = "CLOSE" if remaining_qty == 0.0 else "REDUCE"

        updated = Position(
            symbol=position.symbol,
            quantity=remaining_qty,
            avg_entry_price=position.avg_entry_price,
            current_price=sell_price,
            realized_pnl=position.realized_pnl + realized_pnl,
        )

        self._record(
            symbol=position.symbol,
            action=action,
            quantity_delta=-quantity,
            price=sell_price,
            avg_entry_before=position.avg_entry_price,
            avg_entry_after=position.avg_entry_price,
            remaining_quantity=remaining_qty,
            realized_pnl=realized_pnl,
        )

        log.info(
            "position_reduced",
            symbol=position.symbol,
            sold_qty=quantity,
            sell_price=sell_price,
            realized_pnl=round(realized_pnl, 4),
            remaining_quantity=remaining_qty,
            action=action,
        )
        return updated, realized_pnl

    def get_history(self) -> list[PositionChangeRecord]:
        """Return the full audit trail of position changes (defensive copy)."""
        return list(self._history)

    # ── Internal ────────────────────────────────────────────────

    def _record(
        self,
        *,
        symbol: str,
        action: str,
        quantity_delta: float,
        price: float,
        avg_entry_before: float,
        avg_entry_after: float,
        remaining_quantity: float,
        realized_pnl: float,
    ) -> None:
        """Append an immutable change record to the history."""
        record = PositionChangeRecord(
            record_id=str(uuid7()),
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            action=action,
            quantity_delta=quantity_delta,
            price=price,
            avg_entry_before=avg_entry_before,
            avg_entry_after=avg_entry_after,
            remaining_quantity=remaining_quantity,
            realized_pnl=realized_pnl,
        )
        self._history.append(record)
