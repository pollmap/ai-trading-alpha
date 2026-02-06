"""News and sentiment data adapter using RSS feeds and free APIs."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from config.settings import get_settings
from src.core.logging import get_logger
from src.core.types import NewsItem

log = get_logger(__name__)


class NewsAdapter:
    """Collect news from multiple free sources.

    Sources:
    - Google News RSS (Korean market)
    - Finnhub News API (US market)
    - CryptoPanic API (Crypto market)

    On any error, returns empty list — news absence must never halt trading cycles.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Korean News (Google News RSS) ────────────────────────────

    async def fetch_kr_news(self, query: str = "코스피 주식") -> list[NewsItem]:
        """Fetch Korean market news via Google News RSS."""
        try:
            import feedparser

            url = (
                f"https://news.google.com/rss/search?"
                f"q={query}&hl=ko&gl=KR&ceid=KR:ko"
            )
            client = await self._get_client()
            resp = await client.get(url)
            resp.raise_for_status()

            feed = await asyncio.to_thread(feedparser.parse, resp.text)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=3)
            items: list[NewsItem] = []

            for entry in feed.entries[:20]:
                try:
                    published = datetime(
                        *entry.published_parsed[:6], tzinfo=timezone.utc
                    ) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.now(timezone.utc)

                    if published < cutoff:
                        continue

                    items.append(NewsItem(
                        timestamp=published,
                        title=entry.get("title", ""),
                        summary=entry.get("summary", "")[:500],
                        source="Google News KR",
                        relevance_score=0.5,
                        sentiment=0.0,
                    ))
                except (AttributeError, TypeError, ValueError):
                    continue

            log.info("kr_news_fetched", count=len(items))
            return items

        except Exception as exc:
            log.warning("kr_news_fetch_failed", error=str(exc))
            return []

    # ── US News (Finnhub) ────────────────────────────────────────

    async def fetch_us_news(self, category: str = "general") -> list[NewsItem]:
        """Fetch US market news via Finnhub API."""
        api_key = self._settings.finnhub_api_key.get_secret_value()
        if not api_key:
            log.debug("finnhub_api_key_not_set")
            return []

        try:
            client = await self._get_client()
            resp = await client.get(
                "https://finnhub.io/api/v1/news",
                params={"category": category, "token": api_key},
            )
            resp.raise_for_status()
            data: list[dict[str, Any]] = resp.json()

            cutoff = datetime.now(timezone.utc) - timedelta(hours=3)
            items: list[NewsItem] = []

            for article in data[:20]:
                try:
                    ts = datetime.fromtimestamp(
                        article.get("datetime", 0), tz=timezone.utc
                    )
                    if ts < cutoff:
                        continue

                    items.append(NewsItem(
                        timestamp=ts,
                        title=article.get("headline", ""),
                        summary=article.get("summary", "")[:500],
                        source=article.get("source", "Finnhub"),
                        relevance_score=0.5,
                        sentiment=0.0,
                    ))
                except (TypeError, ValueError):
                    continue

            log.info("us_news_fetched", count=len(items))
            return items

        except Exception as exc:
            log.warning("us_news_fetch_failed", error=str(exc))
            return []

    # ── Crypto News (CryptoPanic) ────────────────────────────────

    async def fetch_crypto_news(self) -> list[NewsItem]:
        """Fetch crypto news via CryptoPanic API."""
        api_key = self._settings.cryptopanic_api_key.get_secret_value()
        if not api_key:
            log.debug("cryptopanic_api_key_not_set")
            return await self._fetch_crypto_news_fallback()

        try:
            client = await self._get_client()
            resp = await client.get(
                "https://cryptopanic.com/api/free/v1/posts/",
                params={
                    "auth_token": api_key,
                    "currencies": "BTC,ETH,SOL",
                    "kind": "news",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            items: list[NewsItem] = []
            for post in data.get("results", [])[:20]:
                try:
                    ts_str = post.get("published_at", "")
                    ts = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00")
                    ) if ts_str else datetime.now(timezone.utc)

                    items.append(NewsItem(
                        timestamp=ts,
                        title=post.get("title", ""),
                        summary=post.get("title", ""),
                        source="CryptoPanic",
                        relevance_score=0.7,
                        sentiment=0.0,
                    ))
                except (TypeError, ValueError):
                    continue

            log.info("crypto_news_fetched", count=len(items))
            return items

        except Exception as exc:
            log.warning("crypto_news_fetch_failed", error=str(exc))
            return []

    async def _fetch_crypto_news_fallback(self) -> list[NewsItem]:
        """Fallback: fetch crypto news from Google News RSS."""
        try:
            import feedparser

            client = await self._get_client()
            resp = await client.get(
                "https://news.google.com/rss/search?q=bitcoin+ethereum+crypto&hl=en&gl=US"
            )
            resp.raise_for_status()

            feed = await asyncio.to_thread(feedparser.parse, resp.text)
            items: list[NewsItem] = []

            for entry in feed.entries[:10]:
                try:
                    published = datetime(
                        *entry.published_parsed[:6], tzinfo=timezone.utc
                    ) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.now(timezone.utc)

                    items.append(NewsItem(
                        timestamp=published,
                        title=entry.get("title", ""),
                        summary=entry.get("summary", "")[:500],
                        source="Google News Crypto",
                        relevance_score=0.3,
                        sentiment=0.0,
                    ))
                except (AttributeError, TypeError, ValueError):
                    continue

            log.info("crypto_news_fallback_fetched", count=len(items))
            return items

        except Exception as exc:
            log.warning("crypto_news_fallback_failed", error=str(exc))
            return []

    # ── Aggregation ──────────────────────────────────────────────

    async def fetch_all(self, market: str) -> list[NewsItem]:
        """Fetch news for a specific market. Returns empty list on failure."""
        if market == "KRX":
            return await self.fetch_kr_news()
        elif market == "US":
            return await self.fetch_us_news()
        elif market == "CRYPTO":
            return await self.fetch_crypto_news()
        return []
