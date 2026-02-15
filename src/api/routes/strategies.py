"""Custom strategy CRUD endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from uuid_extensions import uuid7

from src.api.deps import get_db_engine, require_auth
from src.api.models.schemas import StrategyCreate, StrategyOut, StrategyUpdate
from src.core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.post("", response_model=StrategyOut, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    body: StrategyCreate,
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> StrategyOut:
    """Create a new custom strategy."""
    strategy_id = str(uuid7())
    now = datetime.now(timezone.utc)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO strategies
                    (strategy_id, tenant_id, name, description, prompt_template,
                     model, markets, risk_params, status, version, tags,
                     created_at, updated_at)
                VALUES
                    (:sid, :tid, :name, :desc, :prompt,
                     :model, :markets, :risk, 'draft', 1, :tags,
                     :now, :now)
                """
            ),
            {
                "sid": strategy_id,
                "tid": tenant_id,
                "name": body.name,
                "desc": body.description,
                "prompt": body.prompt_template,
                "model": body.model,
                "markets": json.dumps(body.markets),
                "risk": json.dumps(body.risk_params),
                "tags": json.dumps(body.tags),
                "now": now,
            },
        )

    log.info("strategy_created", strategy_id=strategy_id, tenant_id=tenant_id)

    return StrategyOut(
        strategy_id=strategy_id,
        tenant_id=tenant_id,
        name=body.name,
        description=body.description,
        prompt_template=body.prompt_template,
        model=body.model,
        markets=body.markets,
        risk_params=body.risk_params,
        tags=body.tags,
        created_at=now,
        updated_at=now,
    )


@router.get("", response_model=list[StrategyOut])
async def list_strategies(
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> list[StrategyOut]:
    """List all strategies for the authenticated tenant."""
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT * FROM strategies WHERE tenant_id = :tid "
                "ORDER BY updated_at DESC"
            ),
            {"tid": tenant_id},
        )
        rows = result.mappings().all()

    return [_row_to_strategy(row) for row in rows]


@router.get("/{strategy_id}", response_model=StrategyOut)
async def get_strategy(
    strategy_id: str,
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> StrategyOut:
    """Get a single strategy."""
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT * FROM strategies "
                "WHERE strategy_id = :sid AND tenant_id = :tid"
            ),
            {"sid": strategy_id, "tid": tenant_id},
        )
        row = result.mappings().first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found",
        )

    return _row_to_strategy(row)


@router.patch("/{strategy_id}", response_model=StrategyOut)
async def update_strategy(
    strategy_id: str,
    body: StrategyUpdate,
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> StrategyOut:
    """Update a strategy (partial update)."""
    # Check existence
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT * FROM strategies "
                "WHERE strategy_id = :sid AND tenant_id = :tid"
            ),
            {"sid": strategy_id, "tid": tenant_id},
        )
        existing = result.mappings().first()

    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found",
        )

    # Build dynamic SET clause
    updates: dict[str, object] = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.description is not None:
        updates["description"] = body.description
    if body.prompt_template is not None:
        updates["prompt_template"] = body.prompt_template
    if body.model is not None:
        updates["model"] = body.model
    if body.markets is not None:
        updates["markets"] = json.dumps(body.markets)
    if body.risk_params is not None:
        updates["risk_params"] = json.dumps(body.risk_params)
    if body.tags is not None:
        updates["tags"] = json.dumps(body.tags)
    if body.status is not None:
        updates["status"] = body.status

    if not updates:
        return _row_to_strategy(existing)

    now = datetime.now(timezone.utc)
    updates["updated_at"] = now
    updates["version"] = existing["version"] + 1

    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["sid"] = strategy_id
    updates["tid"] = tenant_id

    async with engine.begin() as conn:
        await conn.execute(
            text(
                f"UPDATE strategies SET {set_clause} "  # noqa: S608
                "WHERE strategy_id = :sid AND tenant_id = :tid"
            ),
            updates,
        )

    # Re-fetch
    return await get_strategy(strategy_id, tenant_id, engine)


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: str,
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> None:
    """Delete a strategy."""
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                "SELECT 1 FROM strategies "
                "WHERE strategy_id = :sid AND tenant_id = :tid"
            ),
            {"sid": strategy_id, "tid": tenant_id},
        )
        if result.first() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found",
            )

        await conn.execute(
            text(
                "DELETE FROM strategies "
                "WHERE strategy_id = :sid AND tenant_id = :tid"
            ),
            {"sid": strategy_id, "tid": tenant_id},
        )


def _row_to_strategy(row: object) -> StrategyOut:
    """Convert DB row to StrategyOut."""
    markets = row["markets"]  # type: ignore[index]
    if isinstance(markets, str):
        markets = json.loads(markets)
    risk_params = row["risk_params"]  # type: ignore[index]
    if isinstance(risk_params, str):
        risk_params = json.loads(risk_params)
    tags = row.get("tags", [])  # type: ignore[union-attr]
    if isinstance(tags, str):
        tags = json.loads(tags)

    return StrategyOut(
        strategy_id=row["strategy_id"],  # type: ignore[index]
        tenant_id=row["tenant_id"],  # type: ignore[index]
        name=row["name"],  # type: ignore[index]
        description=row.get("description", ""),  # type: ignore[union-attr]
        prompt_template=row["prompt_template"],  # type: ignore[index]
        model=row["model"],  # type: ignore[index]
        markets=markets,
        risk_params=risk_params,
        status=row.get("status", "draft"),  # type: ignore[union-attr]
        version=row.get("version", 1),  # type: ignore[union-attr]
        tags=tags,
        created_at=row["created_at"],  # type: ignore[index]
        updated_at=row["updated_at"],  # type: ignore[index]
    )
