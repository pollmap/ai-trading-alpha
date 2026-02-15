"""PostgreSQL + TimescaleDB connection and schema definitions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from config.settings import get_settings
from src.core.logging import get_logger

log = get_logger(__name__)

metadata = MetaData()

# ── Tables ───────────────────────────────────────────────────────

market_snapshots = Table(
    "market_snapshots",
    metadata,
    Column("snapshot_id", String, primary_key=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("market", String, nullable=False, index=True),
    Column("data", JSONB, nullable=False),
)

trading_signals = Table(
    "trading_signals",
    metadata,
    Column("signal_id", String, primary_key=True),
    Column("snapshot_id", String, nullable=False, index=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("model", String, nullable=False, index=True),
    Column("architecture", String, nullable=False, index=True),
    Column("symbol", String, nullable=False),
    Column("action", String, nullable=False),
    Column("weight", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("reasoning", Text, nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("token_usage", JSONB),
)

portfolio_states = Table(
    "portfolio_states",
    metadata,
    Column("portfolio_id", String, nullable=False),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("model", String, nullable=False, index=True),
    Column("architecture", String, nullable=False),
    Column("market", String, nullable=False),
    Column("cash", Float, nullable=False),
    Column("positions", JSONB, nullable=False),
    Column("total_value", Float, nullable=False),
)

trades = Table(
    "trades",
    metadata,
    Column("trade_id", String, primary_key=True),
    Column("signal_id", String, nullable=False, index=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("symbol", String, nullable=False),
    Column("action", String, nullable=False),
    Column("quantity", Float, nullable=False),
    Column("price", Float, nullable=False),
    Column("commission", Float, nullable=False),
    Column("slippage", Float, nullable=False),
    Column("realized_pnl", Float),
)

cost_logs = Table(
    "cost_logs",
    metadata,
    Column("log_id", String, primary_key=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("model", String, nullable=False, index=True),
    Column("input_tokens", Integer, nullable=False),
    Column("output_tokens", Integer, nullable=False),
    Column("cached_tokens", Integer, default=0),
    Column("cost_usd", Float, nullable=False),
    Column("latency_ms", Float, nullable=False),
)

llm_call_logs = Table(
    "llm_call_logs",
    metadata,
    Column("call_id", String, primary_key=True),
    Column("signal_id", String, nullable=True, index=True),
    Column("timestamp", DateTime(timezone=True), nullable=False, index=True),
    Column("model", String, nullable=False, index=True),
    Column("role", String, nullable=False),
    Column("prompt_text", Text, nullable=False),
    Column("raw_response", Text, nullable=False),
    Column("parsed_success", Boolean, nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
)


# ── Engine ───────────────────────────────────────────────────────

_engine: AsyncEngine | None = None


async def get_engine() -> AsyncEngine:
    """Get or create the async database engine (singleton)."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        settings = get_settings()
        db_url = settings.database_url.get_secret_value()
        _engine = create_async_engine(
            db_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
        log.info("database_engine_created", host=db_url.split("@")[-1].split("?")[0])
    return _engine


async def init_schema() -> None:
    """Create all tables and convert to TimescaleDB hypertables."""
    engine = await get_engine()

    async with engine.begin() as conn:
        # Create TimescaleDB extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))

        # Create all tables
        await conn.run_sync(metadata.create_all)

        # Convert time-series tables to hypertables
        hypertables = [
            ("market_snapshots", "timestamp"),
            ("portfolio_states", "timestamp"),
        ]
        for table_name, time_col in hypertables:
            try:
                await conn.execute(
                    text(
                        f"SELECT create_hypertable('{table_name}', '{time_col}', "
                        f"if_not_exists => TRUE, migrate_data => TRUE)"
                    )
                )
                log.info("hypertable_created", table=table_name, time_column=time_col)
            except Exception as exc:
                log.warning("hypertable_creation_skipped", table=table_name, reason=str(exc))

    log.info("schema_initialized")


async def close_engine() -> None:
    """Dispose the database engine."""
    global _engine  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        log.info("database_engine_closed")
