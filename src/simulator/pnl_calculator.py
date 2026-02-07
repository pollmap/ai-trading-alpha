"""Profit-and-loss calculation for portfolios and positions.

All monetary values are in the portfolio's native currency (KRW / USD / USDT).
FX conversion, if needed, happens at a higher layer.
"""

from __future__ import annotations

from src.core.logging import get_logger
from src.core.types import PortfolioState, Position

log = get_logger(__name__)


class PnLCalculator:
    """Stateless calculator for realized / unrealized P&L and portfolio mark-to-market."""

    # ── Realized P&L ────────────────────────────────────────────

    @staticmethod
    def calculate_realized_pnl(
        sell_price: float,
        avg_entry_price: float,
        quantity: float,
        commission: float,
    ) -> float:
        """Calculate net realized P&L for a (partial) position close.

        Args:
            sell_price: Execution price of the sell order.
            avg_entry_price: Weighted-average entry price of the position.
            quantity: Number of units sold.
            commission: Total commission + tax for this sell trade.

        Returns:
            Net realized P&L after deducting *commission*.
        """
        gross_pnl = (sell_price - avg_entry_price) * quantity
        net_pnl = gross_pnl - commission

        log.debug(
            "realized_pnl_calculated",
            sell_price=sell_price,
            avg_entry_price=avg_entry_price,
            quantity=quantity,
            gross_pnl=round(gross_pnl, 4),
            commission=round(commission, 4),
            net_pnl=round(net_pnl, 4),
        )
        return net_pnl

    # ── Unrealized P&L ──────────────────────────────────────────

    @staticmethod
    def calculate_unrealized_pnl(
        positions: dict[str, Position],
        current_prices: dict[str, float],
    ) -> float:
        """Calculate total unrealized P&L across all open positions.

        For symbols missing from *current_prices*, the position's stored
        ``current_price`` is used as a fallback.

        Args:
            positions: Symbol-keyed map of open positions.
            current_prices: Symbol-keyed map of latest market prices.

        Returns:
            Aggregate unrealized P&L.
        """
        total_unrealized = 0.0

        for symbol, position in positions.items():
            if position.quantity <= 0:
                continue
            price = current_prices.get(symbol, position.current_price)
            unrealized = (price - position.avg_entry_price) * position.quantity
            total_unrealized += unrealized

        log.debug(
            "unrealized_pnl_calculated",
            n_positions=len(positions),
            total_unrealized=round(total_unrealized, 4),
        )
        return total_unrealized

    # ── Mark-to-Market ──────────────────────────────────────────

    @staticmethod
    def update_portfolio_values(
        portfolio: PortfolioState,
        current_prices: dict[str, float],
    ) -> PortfolioState:
        """Return a new ``PortfolioState`` with positions marked to current prices.

        Cash, initial capital, and realized P&L are preserved.
        Only ``current_price`` and ``unrealized_pnl`` on each position
        are refreshed.

        Args:
            portfolio: The portfolio to update.
            current_prices: Symbol-keyed map of latest market prices.

        Returns:
            A new ``PortfolioState`` with refreshed valuations.
        """
        updated_positions: dict[str, Position] = {}

        for symbol, position in portfolio.positions.items():
            price = current_prices.get(symbol, position.current_price)
            unrealized = (price - position.avg_entry_price) * position.quantity

            updated_positions[symbol] = Position(
                symbol=position.symbol,
                quantity=position.quantity,
                avg_entry_price=position.avg_entry_price,
                current_price=price,
                unrealized_pnl=unrealized,
                realized_pnl=position.realized_pnl,
            )

        updated = PortfolioState(
            portfolio_id=portfolio.portfolio_id,
            model=portfolio.model,
            architecture=portfolio.architecture,
            market=portfolio.market,
            cash=portfolio.cash,
            positions=updated_positions,
            initial_capital=portfolio.initial_capital,
            created_at=portfolio.created_at,
        )

        log.debug(
            "portfolio_values_updated",
            portfolio_id=portfolio.portfolio_id,
            total_value=round(updated.total_value, 4),
            cash=round(portfolio.cash, 4),
            n_positions=len(updated_positions),
        )
        return updated
