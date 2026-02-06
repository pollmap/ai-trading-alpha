# ATLAS — AI Trading Lab for Agent Strategy

## Project Overview

Benchmark system comparing 4 LLMs (DeepSeek, Gemini, Claude, GPT-4o-mini) x 2 agent architectures (Single, Multi) via real-time virtual trading. Targets 3 markets: Korean stocks (KRX), US stocks, and Cryptocurrency.

## Build & Run

- `make install` — Install dependencies
- `make db-up` — Start PostgreSQL + Redis via docker-compose
- `make db-init` — Create schema
- `make test` — Run all pytest
- `make test-unit` — Unit tests only
- `make run` — Start benchmark
- `make dashboard` — Launch Streamlit dashboard

## Tech Stack

- Python 3.11+, asyncio-based
- LangGraph (multi-agent orchestration)
- PostgreSQL + TimescaleDB (time-series storage)
- Redis (cache + latest prices)
- Streamlit + Plotly (dashboard)

## Absolute Rules (Fix Immediately on Violation)

1. **All API calls must go through cost_tracker middleware**
2. **No hardcoded API keys** — must flow through .env -> config/settings.py
3. **All IO functions must be async/await** — no synchronous blocking calls
4. **Timestamps are always UTC** — convert to KST only for display
5. **LLM parse failure -> return HOLD** — never crash the full cycle on parse error
6. **Each agent combination has independent PortfolioState** — no shared state
7. **MarketSnapshot is immutable** — no modification after creation
8. **All trading signals must have reasoning** — empty string forbidden
9. **Use structlog** — no print(), no stdlib logging module
10. **100% type hints** — Any type forbidden
11. **All LLM calls must be recorded via call_logger** — prompt + raw response stored in DB

## Directory Structure Summary

- `src/core/` — Shared types, interfaces, constants
- `src/data/` — Market data collection & normalization (MIS)
- `src/llm/` — LLM provider adapters
- `src/agents/` — Single/multi agent execution (AEP)
- `src/simulator/` — Virtual trading simulation
- `src/analytics/` — Performance analysis engine (PAI)
- `src/dashboard/` — Streamlit dashboard

## Common Patterns

- Check `src/core/interfaces.py` ABCs before writing new classes
- Load config values from `config/settings.py` (Pydantic Settings)
- Define custom exceptions in `src/core/exceptions.py`
- New module -> corresponding test file in `tests/unit/` mandatory

## Common Pitfalls (Avoid These)

- pykrx is synchronous -> wrap with `asyncio.to_thread()`
- DeepSeek API is OpenAI SDK compatible -> use `openai.AsyncOpenAI(base_url=...)`
- Binance WebSocket disconnects -> auto-reconnect logic mandatory
- LangGraph state uses TypedDict -> NOT dataclass
- Streamlit has its own event loop -> use `nest_asyncio` if asyncio conflicts arise
- GPT-4o-mini adapter is structurally identical to DeepSeek adapter (both use OpenAI SDK)
