"""JSONL-based results store for benchmark persistence.

Stores benchmark results as append-only JSONL files in ``data/results/``.
The dashboard reads from these files to display real data.

File layout::

    data/results/
    ├── status.json           # Current benchmark status (overwritten each cycle)
    ├── equity_curves.jsonl   # Portfolio values over time (appended each cycle)
    ├── signals.jsonl         # All trading signals (appended each cycle)
    ├── trades.jsonl          # All executed trades (appended each cycle)
    └── costs.jsonl           # LLM cost records (appended each cycle)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.core.types import (
    Action,
    AgentArchitecture,
    Market,
    ModelProvider,
    PortfolioState,
    TradingSignal,
)

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "data" / "results"


def _serialize_datetime(obj: Any) -> Any:
    """JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


class ResultsStore:
    """Append-only JSONL results store for benchmark data.

    All write methods append a single JSON line to the appropriate file.
    All read methods return lists of parsed JSON dicts.

    Usage::

        store = ResultsStore()
        store.save_signal(signal, cycle=1)
        store.save_portfolio_snapshot(portfolio, cycle=1)
        store.save_trade(trade_dict, cycle=1)
        store.update_status(status_dict)

        # Dashboard reads:
        signals = store.load_signals()
        curves = store.load_equity_curves()
    """

    def __init__(self, results_dir: Path | None = None) -> None:
        self._dir = results_dir or DEFAULT_RESULTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def results_dir(self) -> Path:
        return self._dir

    # ── Write methods ─────────────────────────────────────────────

    def save_signal(self, signal: TradingSignal, cycle: int) -> None:
        """Append a trading signal record."""
        record = {
            "cycle": cycle,
            "timestamp": signal.timestamp.isoformat(),
            "signal_id": signal.signal_id,
            "model": signal.model.value,
            "architecture": signal.architecture.value,
            "market": signal.market.value,
            "symbol": signal.symbol,
            "action": signal.action.value,
            "weight": round(signal.weight, 4),
            "confidence": round(signal.confidence, 4),
            "reasoning": signal.reasoning,
            "latency_ms": round(signal.latency_ms, 1),
        }
        self._append_jsonl("signals.jsonl", record)

    def save_trade(
        self,
        trade_id: str,
        signal_id: str,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        commission: float,
        realized_pnl: float,
        cycle: int,
    ) -> None:
        """Append a trade record."""
        record = {
            "cycle": cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": trade_id,
            "signal_id": signal_id,
            "symbol": symbol,
            "action": action,
            "quantity": round(quantity, 6),
            "price": round(price, 4),
            "commission": round(commission, 4),
            "realized_pnl": round(realized_pnl, 4),
        }
        self._append_jsonl("trades.jsonl", record)

    def save_portfolio_snapshot(
        self,
        portfolio: PortfolioState,
        cycle: int,
    ) -> None:
        """Append a portfolio equity curve data point."""
        positions_dict: dict[str, dict[str, float]] = {}
        for sym, pos in portfolio.positions.items():
            positions_dict[sym] = {
                "quantity": round(pos.quantity, 6),
                "avg_entry_price": round(pos.avg_entry_price, 4),
                "current_price": round(pos.current_price, 4),
                "unrealized_pnl": round(pos.unrealized_pnl, 4),
            }

        record = {
            "cycle": cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_id": portfolio.portfolio_id,
            "model": portfolio.model.value,
            "architecture": portfolio.architecture.value,
            "market": portfolio.market.value,
            "cash": round(portfolio.cash, 4),
            "total_value": round(portfolio.total_value, 4),
            "initial_capital": portfolio.initial_capital,
            "return_pct": round(
                (portfolio.total_value / portfolio.initial_capital - 1) * 100, 4,
            ) if portfolio.initial_capital > 0 else 0.0,
            "positions": positions_dict,
            "n_positions": len(portfolio.positions),
        }
        self._append_jsonl("equity_curves.jsonl", record)

    def save_cost_record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        cycle: int,
    ) -> None:
        """Append a cost record."""
        record = {
            "cycle": cycle,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 6),
            "latency_ms": round(latency_ms, 1),
        }
        self._append_jsonl("costs.jsonl", record)

    def update_status(
        self,
        running: bool,
        cycle_count: int,
        markets: list[str],
        registered_agents: list[str],
        total_cost_usd: float,
        started_at: str | None = None,
    ) -> None:
        """Overwrite the status file with current benchmark state."""
        status = {
            "running": running,
            "cycle_count": cycle_count,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "started_at": started_at or datetime.now(timezone.utc).isoformat(),
            "markets": markets,
            "registered_agents": registered_agents,
            "total_cost_usd": round(total_cost_usd, 6),
        }
        path = self._dir / "status.json"
        path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    # ── Read methods (for dashboard) ──────────────────────────────

    def load_equity_curves(self) -> list[dict[str, Any]]:
        """Load all equity curve records."""
        return self._read_jsonl("equity_curves.jsonl")

    def load_signals(self) -> list[dict[str, Any]]:
        """Load all signal records."""
        return self._read_jsonl("signals.jsonl")

    def load_trades(self) -> list[dict[str, Any]]:
        """Load all trade records."""
        return self._read_jsonl("trades.jsonl")

    def load_costs(self) -> list[dict[str, Any]]:
        """Load all cost records."""
        return self._read_jsonl("costs.jsonl")

    def load_status(self) -> dict[str, Any] | None:
        """Load current benchmark status."""
        path = self._dir / "status.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def has_data(self) -> bool:
        """Check if any results data exists."""
        return (self._dir / "equity_curves.jsonl").exists()

    # ── Internal ──────────────────────────────────────────────────

    def _append_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        """Append a single JSON line to a file."""
        path = self._dir / filename
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_serialize_datetime) + "\n")

    def _read_jsonl(self, filename: str) -> list[dict[str, Any]]:
        """Read all JSON lines from a file."""
        path = self._dir / filename
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records
