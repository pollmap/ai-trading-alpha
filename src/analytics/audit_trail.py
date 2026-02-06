"""Decision audit trail — query and display agent decision history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.core.logging import get_logger
from src.core.types import TradingSignal

log = get_logger(__name__)


@dataclass
class AuditEntry:
    """Single entry in the decision audit trail."""

    timestamp: datetime
    model: str
    architecture: str
    market: str
    symbol: str
    action: str
    weight: float
    confidence: float
    reasoning: str
    latency_ms: float
    snapshot_id: str


class AuditTrail:
    """Manage and query agent decision history."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, signal: TradingSignal) -> None:
        """Record a trading signal in the audit trail."""
        entry = AuditEntry(
            timestamp=signal.timestamp,
            model=signal.model.value,
            architecture=signal.architecture.value,
            market=signal.market.value,
            symbol=signal.symbol,
            action=signal.action.value,
            weight=signal.weight,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            latency_ms=signal.latency_ms,
            snapshot_id=signal.snapshot_id,
        )
        self._entries.append(entry)

    def query(
        self,
        model: str | None = None,
        architecture: str | None = None,
        market: str | None = None,
        limit: int = 50,
    ) -> list[AuditEntry]:
        """Query audit trail with optional filters."""
        results = self._entries

        if model:
            results = [e for e in results if e.model == model]
        if architecture:
            results = [e for e in results if e.architecture == architecture]
        if market:
            results = [e for e in results if e.market == market]

        return sorted(results, key=lambda e: e.timestamp, reverse=True)[:limit]

    @property
    def total_entries(self) -> int:
        return len(self._entries)
