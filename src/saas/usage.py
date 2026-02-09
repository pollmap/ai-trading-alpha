"""Usage metering and quota enforcement for SaaS tenants.

Tracks:
- API call counts (per day, per month)
- Compute time (simulation runtime)
- LLM token usage and cost
- Storage (portfolios, strategies, reports)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, date

from src.core.logging import get_logger
from src.saas.tenant import Tenant, PLAN_LIMITS

log = get_logger(__name__)


@dataclass
class UsageRecord:
    """Single usage event."""

    tenant_id: str
    event_type: str  # "api_call" | "simulation_cycle" | "llm_call" | "report_export"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    quantity: int = 1
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class UsageSummary:
    """Aggregated usage for a tenant."""

    tenant_id: str
    period: str  # "2026-02-09" or "2026-02"
    api_calls: int = 0
    simulation_cycles: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost_usd: float = 0.0
    reports_exported: int = 0
    active_simulations: int = 0
    active_strategies: int = 0


class UsageMeter:
    """Tracks and enforces usage quotas per tenant."""

    def __init__(self) -> None:
        # tenant_id -> date_str -> event_type -> count
        self._daily: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        self._records: list[UsageRecord] = []

    def record(self, event: UsageRecord) -> None:
        """Record a usage event."""
        day_str = event.timestamp.strftime("%Y-%m-%d")
        self._daily[event.tenant_id][day_str][event.event_type] += event.quantity
        self._records.append(event)

        log.debug(
            "usage_recorded",
            tenant_id=event.tenant_id,
            event_type=event.event_type,
            quantity=event.quantity,
        )

    def check_quota(self, tenant: Tenant, event_type: str) -> bool:
        """Check if the tenant has remaining quota for the event type.

        Returns True if the action is allowed, False if quota exceeded.
        """
        today = date.today().isoformat()
        current = self._daily[tenant.tenant_id][today].get(event_type, 0)

        limits = PLAN_LIMITS[tenant.plan]

        if event_type == "api_call":
            allowed = current < limits["max_api_calls_per_day"]
        elif event_type == "simulation_cycle":
            allowed = True  # Cycles are unlimited within a simulation
        elif event_type == "llm_call":
            allowed = current < limits["max_api_calls_per_day"]
        else:
            allowed = True

        if not allowed:
            log.warning(
                "quota_exceeded",
                tenant_id=tenant.tenant_id,
                event_type=event_type,
                current=current,
                plan=tenant.plan.value,
            )

        return allowed

    def get_daily_summary(self, tenant_id: str, day: str | None = None) -> UsageSummary:
        """Get usage summary for a specific day."""
        if day is None:
            day = date.today().isoformat()

        counts = self._daily[tenant_id][day]
        return UsageSummary(
            tenant_id=tenant_id,
            period=day,
            api_calls=counts.get("api_call", 0),
            simulation_cycles=counts.get("simulation_cycle", 0),
            llm_calls=counts.get("llm_call", 0),
            llm_tokens=counts.get("llm_token", 0),
            reports_exported=counts.get("report_export", 0),
        )

    def get_monthly_summary(self, tenant_id: str, month: str | None = None) -> UsageSummary:
        """Get aggregated usage for a month (format: YYYY-MM)."""
        if month is None:
            month = date.today().strftime("%Y-%m")

        summary = UsageSummary(tenant_id=tenant_id, period=month)
        for day_str, counts in self._daily[tenant_id].items():
            if day_str.startswith(month):
                summary.api_calls += counts.get("api_call", 0)
                summary.simulation_cycles += counts.get("simulation_cycle", 0)
                summary.llm_calls += counts.get("llm_call", 0)
                summary.llm_tokens += counts.get("llm_token", 0)
                summary.reports_exported += counts.get("report_export", 0)

        return summary

    def get_all_records(self, tenant_id: str, limit: int = 100) -> list[UsageRecord]:
        """Get recent usage records for a tenant."""
        tenant_records = [r for r in self._records if r.tenant_id == tenant_id]
        return tenant_records[-limit:]

    def reset_daily(self, tenant_id: str, day: str | None = None) -> None:
        """Reset daily counters (for testing or admin override)."""
        if day is None:
            day = date.today().isoformat()
        if tenant_id in self._daily and day in self._daily[tenant_id]:
            del self._daily[tenant_id][day]
            log.info("usage_reset", tenant_id=tenant_id, day=day)
