"""Redis cache layer for market data."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis

from config.settings import get_settings
from src.core.logging import get_logger
from src.core.types import Market

log = get_logger(__name__)

# TTL per market (seconds)
_CACHE_TTLS: dict[str, int] = {
    "KRX": 60,
    "US": 60,
    "CRYPTO": 10,
}


class RedisCache:
    """Async Redis cache for market snapshots and real-time prices."""

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Initialize the Redis connection."""
        if self._redis is None:
            settings = get_settings()
            redis_url = settings.redis_url.get_secret_value()
            self._redis = aioredis.from_url(
                redis_url,
                decode_responses=True,
            )
            log.info("redis_connected")

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            log.info("redis_closed")

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            await self.connect()
        assert self._redis is not None
        return self._redis

    # ── Snapshot Cache ───────────────────────────────────────────

    async def set_snapshot(self, market: Market, snapshot_data: dict[str, Any]) -> None:
        """Cache a market snapshot with market-specific TTL."""
        r = await self._get_redis()
        key = f"snapshot:{market.value}:latest"
        ttl = _CACHE_TTLS.get(market.value, 60)

        payload = json.dumps(snapshot_data, default=str)
        await r.setex(key, ttl, payload)
        log.debug("snapshot_cached", market=market.value, ttl=ttl)

    async def get_snapshot(self, market: Market) -> dict[str, Any] | None:
        """Retrieve the latest cached snapshot for a market."""
        r = await self._get_redis()
        key = f"snapshot:{market.value}:latest"

        raw = await r.get(key)
        if raw is None:
            return None

        return json.loads(raw)

    # ── Price Cache (for WebSocket real-time prices) ─────────────

    async def set_price(self, symbol: str, price: float) -> None:
        """Cache the latest price for a symbol (used by WebSocket feeds)."""
        r = await self._get_redis()
        key = f"price:{symbol}"
        data = json.dumps({
            "price": price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        await r.setex(key, 30, data)  # 30s TTL for real-time prices

    async def get_price(self, symbol: str) -> float | None:
        """Get the latest cached price for a symbol."""
        r = await self._get_redis()
        key = f"price:{symbol}"

        raw = await r.get(key)
        if raw is None:
            return None

        return json.loads(raw).get("price")

    # ── Benchmark State ──────────────────────────────────────────

    async def set_state(self, key: str, data: dict[str, Any]) -> None:
        """Persist benchmark state (e.g., cycle count, last run time)."""
        r = await self._get_redis()
        await r.set(f"state:{key}", json.dumps(data, default=str))

    async def get_state(self, key: str) -> dict[str, Any] | None:
        """Retrieve benchmark state."""
        r = await self._get_redis()
        raw = await r.get(f"state:{key}")
        if raw is None:
            return None
        return json.loads(raw)

    # ── Health Check ─────────────────────────────────────────────

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            r = await self._get_redis()
            return await r.ping()
        except Exception:
            return False
