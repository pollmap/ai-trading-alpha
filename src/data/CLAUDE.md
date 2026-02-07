# Data Module (Market Intelligence Stream) Rules

## Role

Collect market data from external APIs, normalize, and produce MarketSnapshot objects.

## Rules

- All adapters implement BaseMarketDataAdapter
- External API calls must include try/except + retry logic
- Rate limit exceeded -> exponential backoff (1s, 2s, 4s, max 30s)
- Missing data: forward-fill up to 3 hours, beyond -> exclude symbol + warning log
- Cache: Redis storage, TTL varies by market (KRX: 60s, US: 60s, Crypto: 10s)
- New data source -> create file under adapters/, add mapping in normalizer.py
- pykrx is synchronous -> all calls must use asyncio.to_thread()
- News adapter errors -> return empty list (never halt cycles for missing news)

## API Endpoints Reference

- KIS API base: https://openapi.koreainvestment.com:9443
- KIS mock trading: https://openapivts.koreainvestment.com:29443
- ECOS: https://ecos.bok.or.kr/api/
- OpenDART: https://opendart.fss.or.kr/api/
- FRED: https://api.stlouisfed.org/fred/
- EODHD: https://eodhd.com/api/
- Binance WS: wss://stream.binance.com:9443/ws/
- Finnhub: https://finnhub.io/api/v1/
- CryptoPanic: https://cryptopanic.com/api/free/v1/posts/

## Testing

- Unit: mock API responses for normalizer tests
- Integration: live API calls (tests/integration/test_data_adapters.py)
- Store sample API responses in fixtures/
