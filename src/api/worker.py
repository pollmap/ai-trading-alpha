"""Simulation worker — polls Redis queue and runs BenchmarkRunner.

Usage:
    python -m src.api.worker

Continuously polls the Redis 'atlas:simulation_queue' list (BRPOP)
and executes simulations using the existing BenchmarkRunner.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from sqlalchemy import text

from config.settings import get_settings
from src.core.logging import get_logger
from src.data.db import get_engine

log = get_logger(__name__)

QUEUE_KEY = "atlas:simulation_queue"
POLL_TIMEOUT = 5  # seconds for BRPOP


async def _update_simulation_status(
    simulation_id: str,
    status: str,
    *,
    error_message: str | None = None,
    total_cycles: int = 0,
) -> None:
    """Update simulation status in the database."""
    engine = await get_engine()
    now = datetime.now(timezone.utc)

    params: dict[str, object] = {
        "sid": simulation_id,
        "status": status,
    }

    if status == "running":
        query = (
            "UPDATE simulations SET status = :status, started_at = :now "
            "WHERE simulation_id = :sid"
        )
        params["now"] = now
    elif status in ("completed", "failed"):
        query = (
            "UPDATE simulations SET status = :status, completed_at = :now, "
            "total_cycles = :cycles, error_message = :err "
            "WHERE simulation_id = :sid"
        )
        params["now"] = now
        params["cycles"] = total_cycles
        params["err"] = error_message
    else:
        query = "UPDATE simulations SET status = :status WHERE simulation_id = :sid"

    async with engine.begin() as conn:
        await conn.execute(text(query), params)


async def _get_simulation_config(simulation_id: str) -> dict[str, object] | None:
    """Fetch simulation config from the database."""
    engine = await get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            text("SELECT config_json FROM simulations WHERE simulation_id = :sid"),
            {"sid": simulation_id},
        )
        row = result.mappings().first()
        if row is None:
            return None
        config = row["config_json"]
        if isinstance(config, str):
            return json.loads(config)  # type: ignore[no-any-return]
        return config  # type: ignore[return-value]


async def _run_simulation(simulation_id: str, tenant_id: str) -> None:
    """Execute a single simulation job."""
    log.info("simulation_starting", simulation_id=simulation_id, tenant_id=tenant_id)

    await _update_simulation_status(simulation_id, "running")

    config = await _get_simulation_config(simulation_id)
    if config is None:
        log.error("simulation_config_missing", simulation_id=simulation_id)
        await _update_simulation_status(
            simulation_id, "failed", error_message="Config not found"
        )
        return

    try:
        # Import here to avoid circular imports at module level
        from scripts.run_benchmark import BenchmarkRunner

        markets = [str(m) for m in config.get("markets", ["US"])]
        cycles = int(config.get("cycles", 10))

        runner = BenchmarkRunner(markets=markets, max_cycles=cycles)
        await runner.run()

        await _update_simulation_status(
            simulation_id, "completed", total_cycles=cycles
        )
        log.info("simulation_completed", simulation_id=simulation_id)

    except Exception as exc:
        log.error(
            "simulation_failed",
            simulation_id=simulation_id,
            error=str(exc),
        )
        await _update_simulation_status(
            simulation_id, "failed", error_message=str(exc)[:500]
        )


async def worker_loop() -> None:
    """Main worker loop — poll Redis and process jobs."""
    import redis.asyncio as aioredis

    settings = get_settings()
    redis_client = aioredis.from_url(settings.redis_url.get_secret_value())

    log.info("worker_started", queue=QUEUE_KEY)

    try:
        while True:
            # BRPOP blocks for POLL_TIMEOUT seconds
            result = await redis_client.brpop(QUEUE_KEY, timeout=POLL_TIMEOUT)
            if result is None:
                continue  # timeout, poll again

            _, raw_job = result
            try:
                job = json.loads(raw_job)
                simulation_id = job["simulation_id"]
                tenant_id = job["tenant_id"]
            except (json.JSONDecodeError, KeyError) as exc:
                log.warning("invalid_job", raw=str(raw_job), error=str(exc))
                continue

            await _run_simulation(simulation_id, tenant_id)

    except asyncio.CancelledError:
        log.info("worker_cancelled")
    finally:
        await redis_client.aclose()
        log.info("worker_stopped")


if __name__ == "__main__":
    asyncio.run(worker_loop())
