"""Portfolio manager — creates and maintains independent portfolio state.

Manages 9 portfolios per market:
    4 LLM providers  x  2 agent architectures  =  8 agent portfolios
    + 1 buy-and-hold benchmark
    ──────────────────────────────────────────────
    = 9 portfolios per market

The buy-and-hold portfolio uses ``ModelProvider.DEEPSEEK`` /
``AgentArchitecture.SINGLE`` as placeholder enum values.  Always access
it via :meth:`get_buy_hold_state` rather than :meth:`get_state` so the
intent is explicit.
"""

from __future__ import annotations

from datetime import datetime, timezone

from uuid_extensions import uuid7

from src.core.exceptions import InsufficientFundsError
from src.core.logging import get_logger
from src.core.types import (
    AgentArchitecture,
    Market,
    ModelProvider,
    PortfolioState,
    Position,
)
from src.simulator.position_tracker import PositionTracker

log = get_logger(__name__)

# Placeholder enum values for the buy-and-hold benchmark portfolio.
# The buy-and-hold portfolio is not tied to any real model/architecture;
# these are required solely to satisfy the PortfolioState dataclass.
_BUY_HOLD_MODEL = ModelProvider.DEEPSEEK
_BUY_HOLD_ARCH = AgentArchitecture.SINGLE


class PortfolioManager:
    """Central registry for all portfolio state across markets.

    Typical lifecycle::

        mgr = PortfolioManager()
        mgr.init_portfolios(Market.KRX, 100_000_000)
        mgr.init_portfolios(Market.US, 100_000)
        mgr.init_portfolios(Market.CRYPTO, 100_000)

        state = mgr.get_state(ModelProvider.CLAUDE, AgentArchitecture.MULTI, Market.US)
        mgr.update_position(state.portfolio_id, "AAPL", 10, 185.50)

        all_states = mgr.snapshot_all()
    """

    def __init__(self) -> None:
        # Primary store: portfolio_id -> PortfolioState
        self._portfolios: dict[str, PortfolioState] = {}

        # Lookup indices
        self._agent_index: dict[
            tuple[ModelProvider, AgentArchitecture, Market], str
        ] = {}
        self._buy_hold_index: dict[Market, str] = {}

        # Audit trail: one tracker per portfolio
        self._trackers: dict[str, PositionTracker] = {}

    # ── Initialization ──────────────────────────────────────────

    def init_portfolios(
        self,
        market: Market,
        initial_capital: float,
    ) -> list[str]:
        """Create 9 fresh portfolios for *market*.

        If portfolios already exist for this market they are **replaced**.

        Args:
            market: Target market (KRX / US / CRYPTO).
            initial_capital: Starting cash balance for every portfolio.

        Returns:
            List of the 9 newly created portfolio IDs.
        """
        created_ids: list[str] = []

        # 8 agent portfolios: 4 models x 2 architectures
        for model in ModelProvider:
            for arch in AgentArchitecture:
                pid = str(uuid7())
                portfolio = PortfolioState(
                    portfolio_id=pid,
                    model=model,
                    architecture=arch,
                    market=market,
                    cash=initial_capital,
                    positions={},
                    initial_capital=initial_capital,
                    created_at=datetime.now(timezone.utc),
                )
                self._portfolios[pid] = portfolio
                self._agent_index[(model, arch, market)] = pid
                self._trackers[pid] = PositionTracker()
                created_ids.append(pid)

        # 1 buy-and-hold benchmark
        bh_pid = str(uuid7())
        bh_portfolio = PortfolioState(
            portfolio_id=bh_pid,
            model=_BUY_HOLD_MODEL,
            architecture=_BUY_HOLD_ARCH,
            market=market,
            cash=initial_capital,
            positions={},
            initial_capital=initial_capital,
            created_at=datetime.now(timezone.utc),
        )
        self._portfolios[bh_pid] = bh_portfolio
        self._buy_hold_index[market] = bh_pid
        self._trackers[bh_pid] = PositionTracker()
        created_ids.append(bh_pid)

        log.info(
            "portfolios_initialized",
            market=market.value,
            initial_capital=initial_capital,
            n_portfolios=len(created_ids),
            portfolio_ids=created_ids,
        )
        return created_ids

    # ── Accessors ───────────────────────────────────────────────

    def get_state(
        self,
        model: ModelProvider,
        architecture: AgentArchitecture,
        market: Market,
    ) -> PortfolioState:
        """Return the current state for an agent portfolio.

        Args:
            model: LLM provider.
            architecture: Agent architecture (single / multi).
            market: Target market.

        Returns:
            Current ``PortfolioState``.

        Raises:
            KeyError: If the combination has not been initialized.
        """
        key = (model, architecture, market)
        pid = self._agent_index.get(key)
        if pid is None:
            msg = (
                f"No portfolio for {model.value}/{architecture.value}/{market.value}. "
                f"Call init_portfolios() first."
            )
            raise KeyError(msg)
        return self._portfolios[pid]

    def get_buy_hold_state(self, market: Market) -> PortfolioState:
        """Return the buy-and-hold benchmark portfolio for *market*.

        Raises:
            KeyError: If the market has not been initialized.
        """
        pid = self._buy_hold_index.get(market)
        if pid is None:
            msg = (
                f"No buy-and-hold portfolio for {market.value}. "
                f"Call init_portfolios() first."
            )
            raise KeyError(msg)
        return self._portfolios[pid]

    def get_portfolio_by_id(self, portfolio_id: str) -> PortfolioState:
        """Return a portfolio by its unique ID.

        Raises:
            KeyError: If *portfolio_id* is unknown.
        """
        portfolio = self._portfolios.get(portfolio_id)
        if portfolio is None:
            msg = f"Unknown portfolio_id: {portfolio_id}"
            raise KeyError(msg)
        return portfolio

    def get_tracker(self, portfolio_id: str) -> PositionTracker:
        """Return the position-change tracker for a portfolio.

        Raises:
            KeyError: If *portfolio_id* is unknown.
        """
        tracker = self._trackers.get(portfolio_id)
        if tracker is None:
            msg = f"No tracker for portfolio_id: {portfolio_id}"
            raise KeyError(msg)
        return tracker

    # ── Position Updates ────────────────────────────────────────

    def update_position(
        self,
        portfolio_id: str,
        symbol: str,
        quantity_delta: float,
        price: float,
        commission: float = 0.0,
    ) -> PortfolioState:
        """Apply a position change and adjust cash accordingly.

        Positive *quantity_delta* means **buy**; negative means **sell**.

        Cash impact::

            BUY  : cash -= (price * quantity_delta + commission)
            SELL : cash += (price * |quantity_delta| - commission)

        Args:
            portfolio_id: Target portfolio.
            symbol: Ticker / asset identifier.
            quantity_delta: Signed change in units.
            price: Execution price per unit.
            commission: Total commission + tax for this trade (>= 0).

        Returns:
            The updated ``PortfolioState``.

        Raises:
            KeyError: If *portfolio_id* is unknown.
            InsufficientFundsError: If a buy would produce negative cash.
            ValueError: If *quantity_delta* is zero or *price* <= 0.
        """
        if quantity_delta == 0.0:
            msg = "quantity_delta must not be zero"
            raise ValueError(msg)
        if price <= 0:
            msg = f"price must be positive, got {price}"
            raise ValueError(msg)
        if commission < 0:
            msg = f"commission must be non-negative, got {commission}"
            raise ValueError(msg)

        portfolio = self.get_portfolio_by_id(portfolio_id)
        tracker = self._trackers[portfolio_id]
        positions = dict(portfolio.positions)  # shallow copy for mutation

        if quantity_delta > 0:
            # ── BUY ─────────────────────────────────────────────
            total_cost = price * quantity_delta + commission
            if total_cost > portfolio.cash:
                raise InsufficientFundsError(
                    f"Need {total_cost:.2f} but only {portfolio.cash:.2f} available",
                    context={
                        "portfolio_id": portfolio_id,
                        "symbol": symbol,
                        "total_cost": total_cost,
                        "cash": portfolio.cash,
                    },
                )
            new_cash = portfolio.cash - total_cost

            if symbol in positions:
                positions[symbol] = tracker.add_to_position(
                    positions[symbol], quantity_delta, price,
                )
            else:
                positions[symbol] = tracker.open_position(
                    symbol, quantity_delta, price,
                )
        else:
            # ── SELL ────────────────────────────────────────────
            sell_qty = abs(quantity_delta)
            proceeds = price * sell_qty - commission
            new_cash = portfolio.cash + proceeds

            if symbol not in positions:
                msg = f"Cannot sell {symbol}: no position in portfolio {portfolio_id}"
                raise ValueError(msg)

            updated_pos, _realized = tracker.reduce_position(
                positions[symbol], sell_qty, price,
            )
            if updated_pos.quantity == 0.0:
                del positions[symbol]
            else:
                positions[symbol] = updated_pos

        # Build updated state
        updated = PortfolioState(
            portfolio_id=portfolio.portfolio_id,
            model=portfolio.model,
            architecture=portfolio.architecture,
            market=portfolio.market,
            cash=new_cash,
            positions=positions,
            initial_capital=portfolio.initial_capital,
            created_at=portfolio.created_at,
        )
        self._portfolios[portfolio_id] = updated

        log.info(
            "position_updated",
            portfolio_id=portfolio_id,
            symbol=symbol,
            quantity_delta=quantity_delta,
            price=price,
            commission=commission,
            new_cash=round(new_cash, 4),
            total_value=round(updated.total_value, 4),
        )
        return updated

    # ── Snapshots ───────────────────────────────────────────────

    def set_state(self, portfolio_id: str, new_state: PortfolioState) -> None:
        """Replace a portfolio's state with a new snapshot.

        Used by the PnL mark-to-market step to store updated valuations
        without breaking encapsulation.

        Raises:
            KeyError: If *portfolio_id* is not tracked.
        """
        if portfolio_id not in self._portfolios:
            msg = f"Unknown portfolio_id: {portfolio_id}"
            raise KeyError(msg)
        self._portfolios[portfolio_id] = new_state

    def snapshot_all(self) -> list[PortfolioState]:
        """Return a list of every tracked portfolio state.

        The returned list is a shallow copy; modifying it does not
        affect internal state.
        """
        return list(self._portfolios.values())
