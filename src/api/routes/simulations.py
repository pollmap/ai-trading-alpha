"""Simulation endpoints — CRUD + Redis queue submission."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from uuid_extensions import uuid7

from config.settings import get_settings
from src.api.db.tenants import TenantRepository
from src.api.deps import get_db_engine, require_auth
from src.api.models.schemas import SimulationCreate, SimulationOut, SimulationStatus
from src.core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/simulations", tags=["simulations"])


@router.post("", response_model=SimulationOut, status_code=status.HTTP_201_CREATED)
async def create_simulation(
    body: SimulationCreate,
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> SimulationOut:
    """Create a new simulation and enqueue it for async execution."""
    # Enforce tenant quota limits
    repo = TenantRepository(engine)
    tenant = await repo.find_by_id(tenant_id)
    if tenant is not None:
        limits = tenant.limits
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    "SELECT count(*) AS cnt FROM simulations "
                    "WHERE tenant_id = :tid AND status IN ('queued', 'running')"
                ),
                {"tid": tenant_id},
            )
            concurrent = result.scalar() or 0
        if concurrent >= limits.get("max_concurrent_sims", 1):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Concurrent simulation limit reached for your plan",
            )

    sim_id = str(uuid7())
    now = datetime.now(timezone.utc)

    config = {
        "markets": body.markets,
        "models": body.models,
        "architectures": body.architectures,
        "cycles": body.cycles,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO simulations
                    (simulation_id, tenant_id, name, config_json, status, created_at)
                VALUES
                    (:sid, :tid, :name, :config, 'queued', :now)
                """
            ),
            {
                "sid": sim_id,
                "tid": tenant_id,
                "name": body.name,
                "config": json.dumps(config),
                "now": now,
            },
        )

    # Enqueue to Redis
    try:
        import redis.asyncio as aioredis

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url.get_secret_value())
        job = json.dumps({"simulation_id": sim_id, "tenant_id": tenant_id})
        await r.lpush("atlas:simulation_queue", job)
        await r.aclose()
        log.info("simulation_queued", simulation_id=sim_id, tenant_id=tenant_id)
    except Exception as exc:
        log.warning("redis_enqueue_failed", error=str(exc))
        # Update status to created (not queued) if Redis is unavailable
        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE simulations SET status = 'created' WHERE simulation_id = :sid"),
                {"sid": sim_id},
            )

    return SimulationOut(
        simulation_id=sim_id,
        tenant_id=tenant_id,
        name=body.name,
        config_json=config,
        status=SimulationStatus.QUEUED,
        created_at=now,
    )


@router.get("", response_model=list[SimulationOut])
async def list_simulations(
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> list[SimulationOut]:
    """List all simulations for the authenticated tenant."""
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT * FROM simulations WHERE tenant_id = :tid "
                "ORDER BY created_at DESC LIMIT 100"
            ),
            {"tid": tenant_id},
        )
        rows = result.mappings().all()

    return [_row_to_sim(row) for row in rows]


@router.get("/{simulation_id}", response_model=SimulationOut)
async def get_simulation(
    simulation_id: str,
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> SimulationOut:
    """Get a single simulation by ID (tenant-scoped)."""
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT * FROM simulations "
                "WHERE simulation_id = :sid AND tenant_id = :tid"
            ),
            {"sid": simulation_id, "tid": tenant_id},
        )
        row = result.mappings().first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation not found",
        )

    return _row_to_sim(row)


@router.delete("/{simulation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_simulation(
    simulation_id: str,
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> None:
    """Delete a simulation (only if not running).

    Uses a single transaction to avoid TOCTOU race conditions.
    """
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT status FROM simulations "
                "WHERE simulation_id = :sid AND tenant_id = :tid "
                "FOR UPDATE"
            ),
            {"sid": simulation_id, "tid": tenant_id},
        )
        row = result.mappings().first()

        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Simulation not found",
            )

        if row["status"] == "running":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete a running simulation",
            )

        await conn.execute(
            text(
                "DELETE FROM simulations "
                "WHERE simulation_id = :sid AND tenant_id = :tid"
            ),
            {"sid": simulation_id, "tid": tenant_id},
        )


def _row_to_sim(row: object) -> SimulationOut:
    """Convert DB row to SimulationOut."""
    config = row["config_json"]  # type: ignore[index]
    if isinstance(config, str):
        config = json.loads(config)

    return SimulationOut(
        simulation_id=row["simulation_id"],  # type: ignore[index]
        tenant_id=row["tenant_id"],  # type: ignore[index]
        name=row["name"],  # type: ignore[index]
        config_json=config,
        status=SimulationStatus(row["status"]),  # type: ignore[index]
        started_at=row.get("started_at"),  # type: ignore[union-attr]
        completed_at=row.get("completed_at"),  # type: ignore[union-attr]
        total_cycles=row.get("total_cycles", 0),  # type: ignore[union-attr]
        error_message=row.get("error_message"),  # type: ignore[union-attr]
        created_at=row["created_at"],  # type: ignore[index]
    )
