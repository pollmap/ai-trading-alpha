"""Market data collection scheduler with event-driven triggers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Awaitable

from src.core.logging import get_logger
from src.core.types import Market

log = get_logger(__name__)


class EventTrigger:
    """Monitors market conditions and triggers urgent cycles.

    Conditions:
    1. Price spike: >=3% change from last snapshot close
    2. VIX surge: >=20% increase from previous day
    3. FX spike: USD/KRW >=1% change
    4. Extreme sentiment: Crypto Fear&Greed <20 or >80

    Cooldown: 15 minutes per market after trigger.
    """

    def __init__(
        self,
        price_threshold: float = 0.03,
        vix_threshold: float = 0.20,
        fx_threshold: float = 0.01,
        fear_greed_low: int = 20,
        fear_greed_high: int = 80,
        cooldown_minutes: int = 15,
    ) -> None:
        self._price_threshold = price_threshold
        self._vix_threshold = vix_threshold
        self._fx_threshold = fx_threshold
        self._fear_greed_low = fear_greed_low
        self._fear_greed_high = fear_greed_high
        self._cooldown = timedelta(minutes=cooldown_minutes)
        self._last_triggers: dict[str, datetime] = {}
        self._last_prices: dict[str, float] = {}
        self._last_vix: float | None = None
        self._last_usdkrw: float | None = None

    def check_price_trigger(self, symbol: str, current_price: float) -> bool:
        """Check if price changed enough to trigger an emergency cycle."""
        last = self._last_prices.get(symbol)
        self._last_prices[symbol] = current_price

        if last is None or last == 0:
            return False

        change = abs(current_price - last) / last
        if change >= self._price_threshold:
            log.info(
                "event_trigger_price",
                symbol=symbol,
                change_pct=f"{change * 100:.2f}%",
                last=last,
                current=current_price,
            )
            return True
        return False

    def check_vix_trigger(self, current_vix: float) -> bool:
        """Check if VIX surged enough to trigger."""
        last = self._last_vix
        self._last_vix = current_vix

        if last is None or last == 0:
            return False

        change = (current_vix - last) / last
        if change >= self._vix_threshold:
            log.info("event_trigger_vix", change_pct=f"{change * 100:.2f}%")
            return True
        return False

    def check_fx_trigger(self, current_usdkrw: float) -> bool:
        """Check if USD/KRW changed enough to trigger."""
        last = self._last_usdkrw
        self._last_usdkrw = current_usdkrw

        if last is None or last == 0:
            return False

        change = abs(current_usdkrw - last) / last
        if change >= self._fx_threshold:
            log.info("event_trigger_fx", change_pct=f"{change * 100:.2f}%")
            return True
        return False

    def check_fear_greed_trigger(self, index: float) -> bool:
        """Check if Fear & Greed is at extreme levels."""
        if index <= self._fear_greed_low or index >= self._fear_greed_high:
            log.info("event_trigger_fear_greed", index=index)
            return True
        return False

    def is_on_cooldown(self, market: str) -> bool:
        """Check if market is still in cooldown from last trigger."""
        last = self._last_triggers.get(market)
        if last is None:
            return False
        return datetime.now(timezone.utc) - last < self._cooldown

    def record_trigger(self, market: str) -> None:
        """Record that a trigger fired for cooldown tracking."""
        self._last_triggers[market] = datetime.now(timezone.utc)


class MarketScheduler:
    """Schedule market data collection cycles.

    Manages fixed-interval cycles per market and integrates
    with EventTrigger for urgent cycles.
    """

    def __init__(
        self,
        cycle_callback: Callable[[Market], Awaitable[None]],
        intervals: dict[str, int] | None = None,
    ) -> None:
        """
        Args:
            cycle_callback: Async function called each cycle with the market.
            intervals: Market -> interval in minutes. Defaults loaded from config.
        """
        self._callback = cycle_callback
        self._intervals = intervals or {
            "KRX": 30,
            "US": 30,
            "CRYPTO": 15,
        }
        self._event_trigger = EventTrigger()
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        """Start all market scheduling loops."""
        self._running = True
        for market_key, interval in self._intervals.items():
            market = Market(market_key)
            task = asyncio.create_task(
                self._market_loop(market, interval),
                name=f"scheduler_{market_key}",
            )
            self._tasks.append(task)
            log.info("scheduler_started", market=market_key, interval_min=interval)

    async def stop(self) -> None:
        """Stop all scheduling loops gracefully."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        log.info("scheduler_stopped")

    async def trigger_urgent_cycle(self, market: Market) -> None:
        """Trigger an emergency cycle if not on cooldown."""
        market_key = market.value
        if self._event_trigger.is_on_cooldown(market_key):
            log.debug("urgent_cycle_skipped_cooldown", market=market_key)
            return

        self._event_trigger.record_trigger(market_key)
        log.info("urgent_cycle_triggered", market=market_key)

        try:
            await self._callback(market)
        except Exception as exc:
            log.error("urgent_cycle_failed", market=market_key, error=str(exc))

    @property
    def event_trigger(self) -> EventTrigger:
        return self._event_trigger

    async def _market_loop(self, market: Market, interval_minutes: int) -> None:
        """Run fixed-interval cycles for a single market."""
        interval_seconds = interval_minutes * 60

        while self._running:
            try:
                log.info("cycle_starting", market=market.value)
                await self._callback(market)
                log.info("cycle_completed", market=market.value)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("cycle_failed", market=market.value, error=str(exc))

            try:
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
