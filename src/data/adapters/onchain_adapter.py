"""On-chain analytics adapter — whale tracking, exchange flows, funding rates."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import aiohttp

from src.core.logging import get_logger
from src.core.interfaces import DataAdapter

log = get_logger(__name__)


class FlowDirection(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    NEUTRAL = "neutral"


@dataclass
class WhaleTransaction:
    """Record of a large on-chain transfer."""
    tx_hash: str
    timestamp: datetime
    from_label: str  # "exchange" | "whale" | "unknown"
    to_label: str
    amount_usd: float
    token: str
    direction: FlowDirection


@dataclass
class ExchangeFlow:
    """Net exchange inflow/outflow for a token."""
    token: str
    timestamp: datetime
    inflow_usd: float
    outflow_usd: float
    net_flow_usd: float  # positive = net inflow (bearish), negative = net outflow (bullish)
    exchange: str = "aggregate"


@dataclass
class FundingRate:
    """Perpetual futures funding rate snapshot."""
    symbol: str
    timestamp: datetime
    rate: float  # positive = longs pay shorts (bullish sentiment), negative = shorts pay longs
    open_interest_usd: float = 0.0
    exchange: str = "aggregate"


@dataclass
class OnChainSnapshot:
    """Aggregated on-chain data for a single cycle."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    whale_transactions: list[WhaleTransaction] = field(default_factory=list)
    exchange_flows: list[ExchangeFlow] = field(default_factory=list)
    funding_rates: list[FundingRate] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def net_exchange_flow_usd(self) -> float:
        return sum(f.net_flow_usd for f in self.exchange_flows)

    @property
    def avg_funding_rate(self) -> float:
        if not self.funding_rates:
            return 0.0
        return sum(f.rate for f in self.funding_rates) / len(self.funding_rates)

    @property
    def whale_buy_pressure(self) -> float:
        """Ratio of outflows (buying) to total whale volume. Higher = more bullish."""
        outflows = sum(w.amount_usd for w in self.whale_transactions if w.direction == FlowDirection.OUTFLOW)
        total = sum(w.amount_usd for w in self.whale_transactions)
        if total <= 0:
            return 0.5
        return outflows / total

    def to_prompt_line(self) -> str:
        """Compact summary for LLM prompt injection."""
        parts: list[str] = ["[On-Chain]"]
        if self.exchange_flows:
            net = self.net_exchange_flow_usd
            direction = "inflow" if net > 0 else "outflow"
            parts.append(f"Net Exchange {direction}: ${abs(net):,.0f}")
        if self.funding_rates:
            avg_fr = self.avg_funding_rate
            parts.append(f"Avg Funding: {avg_fr:+.4f}")
        if self.whale_transactions:
            parts.append(f"Whale txns: {len(self.whale_transactions)}")
            parts.append(f"Whale buy pressure: {self.whale_buy_pressure:.0%}")
        return " | ".join(parts)


class OnChainAdapter:
    """Fetch on-chain data from public APIs.

    In production, this would connect to:
    - Glassnode / CryptoQuant for exchange flows
    - Whale Alert / Etherscan for whale tracking
    - Binance/Bybit APIs for funding rates

    Currently provides a structured interface with fallback to mock data
    when API keys are unavailable.
    """

    def __init__(
        self,
        whale_alert_api_key: str = "",
        cryptoquant_api_key: str = "",
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._whale_key: str = whale_alert_api_key
        self._cq_key: str = cryptoquant_api_key
        self._session: aiohttp.ClientSession | None = session
        self._owns_session: bool = False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def fetch_funding_rates(self, symbols: list[str]) -> list[FundingRate]:
        """Fetch current funding rates from Binance Futures public API."""
        rates: list[FundingRate] = []
        try:
            session = await self._get_session()
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.warning("funding_rate_fetch_failed", status=resp.status)
                    return rates
                data = await resp.json()
                symbol_set = set(symbols)
                now = datetime.now(timezone.utc)
                for item in data:
                    sym = item.get("symbol", "")
                    if sym in symbol_set:
                        rates.append(FundingRate(
                            symbol=sym,
                            timestamp=now,
                            rate=float(item.get("lastFundingRate", 0.0)),
                            open_interest_usd=0.0,
                            exchange="binance",
                        ))
        except Exception as exc:
            log.warning("funding_rate_fetch_error", error=str(exc))
        log.info("funding_rates_fetched", count=len(rates))
        return rates

    async def fetch_snapshot(self, symbols: list[str]) -> OnChainSnapshot:
        """Fetch all on-chain data for the given symbols."""
        funding_rates = await self.fetch_funding_rates(symbols)
        snapshot = OnChainSnapshot(
            timestamp=datetime.now(timezone.utc),
            funding_rates=funding_rates,
        )
        log.info(
            "onchain_snapshot_fetched",
            funding_count=len(funding_rates),
            whale_count=len(snapshot.whale_transactions),
            flow_count=len(snapshot.exchange_flows),
        )
        return snapshot
