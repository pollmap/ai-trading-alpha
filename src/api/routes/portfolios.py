"""Portfolio endpoints — tenant-scoped read access."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.api.deps import get_db_engine, require_auth
from src.api.models.schemas import PortfolioOut, PositionOut

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


@router.get("", response_model=list[PortfolioOut])
async def list_portfolios(
    tenant_id: str = Depends(require_auth),
    market: str | None = None,
    engine: AsyncEngine = Depends(get_db_engine),
) -> list[PortfolioOut]:
    """List all portfolios for the authenticated tenant."""
    query = "SELECT * FROM portfolios WHERE tenant_id = :tid"
    params: dict[str, str] = {"tid": tenant_id}

    if market:
        query += " AND market = :market"
        params["market"] = market

    query += " ORDER BY created_at DESC"

    async with engine.begin() as conn:
        result = await conn.execute(text(query), params)
        rows = result.mappings().all()

    portfolios: list[PortfolioOut] = []
    for row in rows:
        # Fetch positions for each portfolio
        async with engine.begin() as conn:
            pos_result = await conn.execute(
                text("SELECT * FROM positions WHERE portfolio_id = :pid"),
                {"pid": row["portfolio_id"]},
            )
            pos_rows = pos_result.mappings().all()

        positions = [
            PositionOut(
                symbol=p["symbol"],
                quantity=p["quantity"],
                avg_entry_price=p["avg_entry_price"],
                current_price=p["current_price"],
                unrealized_pnl=(p["current_price"] - p["avg_entry_price"]) * p["quantity"],
            )
            for p in pos_rows
        ]

        total_value = row["cash"] + sum(
            p.current_price * p.quantity for p in positions
        )

        portfolios.append(
            PortfolioOut(
                portfolio_id=row["portfolio_id"],
                model=row["model"],
                architecture=row["architecture"],
                market=row["market"],
                cash=row["cash"],
                total_value=total_value,
                positions=positions,
            )
        )

    return portfolios
