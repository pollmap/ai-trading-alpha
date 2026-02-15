"""Usage metering endpoint — tenant-scoped quota information."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.api.db.tenants import TenantRepository
from src.api.deps import get_db_engine, require_auth
from src.api.models.schemas import UsageOut

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("", response_model=UsageOut)
async def get_usage(
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> UsageOut:
    """Get current usage and limits for the authenticated tenant."""
    repo = TenantRepository(engine)
    tenant = await repo.find_by_id(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Count current usage
    current: dict[str, int] = {}
    async with engine.begin() as conn:
        # Simulations count
        result = await conn.execute(
            text("SELECT count(*) AS cnt FROM simulations WHERE tenant_id = :tid"),
            {"tid": tenant_id},
        )
        current["simulations"] = result.scalar() or 0

        # Strategies count
        result = await conn.execute(
            text("SELECT count(*) AS cnt FROM strategies WHERE tenant_id = :tid"),
            {"tid": tenant_id},
        )
        current["strategies"] = result.scalar() or 0

        # API calls today
        result = await conn.execute(
            text(
                "SELECT COALESCE(SUM(quantity), 0) AS cnt FROM usage_events "
                "WHERE tenant_id = :tid "
                "AND event_type = 'api_call' "
                "AND timestamp >= CURRENT_DATE"
            ),
            {"tid": tenant_id},
        )
        current["api_calls_today"] = result.scalar() or 0

        # Running simulations
        result = await conn.execute(
            text(
                "SELECT count(*) AS cnt FROM simulations "
                "WHERE tenant_id = :tid AND status = 'running'"
            ),
            {"tid": tenant_id},
        )
        current["concurrent_sims"] = result.scalar() or 0

    return UsageOut(
        tenant_id=tenant_id,
        plan=tenant.plan.value,
        limits=tenant.limits,
        current=current,
    )
