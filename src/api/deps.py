"""FastAPI dependency injection — shared instances for routes."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncEngine

from src.api.db.tenants import TenantRepository
from src.api.middleware import get_current_user
from src.data.db import get_engine

# ── Database engine ───────────────────────────────────────────────


async def get_db_engine() -> AsyncEngine:
    """Provide the async database engine."""
    return await get_engine()


# ── Repositories ──────────────────────────────────────────────────


async def get_tenant_repo(
    engine: AsyncEngine = Depends(get_db_engine),
) -> TenantRepository:
    """Provide a TenantRepository instance."""
    return TenantRepository(engine)


# ── Auth dependency ───────────────────────────────────────────────


async def require_auth(
    user: dict[str, str] = Depends(get_current_user),
    engine: AsyncEngine = Depends(get_db_engine),
) -> str:
    """Return the tenant_id of the authenticated user.

    Also verifies that the tenant is still active in the database,
    preventing deactivated users from retaining API access via stale JWTs.
    """
    tenant_id = user["sub"]

    repo = TenantRepository(engine)
    if not await repo.is_active(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account deactivated",
        )

    return tenant_id
