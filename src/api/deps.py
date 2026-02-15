"""FastAPI dependency injection — shared instances for routes."""

from __future__ import annotations

from fastapi import Depends
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
) -> str:
    """Return the tenant_id of the authenticated user."""
    return user["sub"]
