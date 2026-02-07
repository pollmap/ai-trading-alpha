# ATLAS - Development Roadmap & Milestones

## Overview

- **Total timeline:** 5 weeks (20~25 hours/week)
- **Week 1:** Foundation + Core + DB
- **Week 2:** Market Data Pipeline (MIS)
- **Week 3:** LLM Adapters + Single Agent
- **Week 4:** Multi-Agent Pipeline + Simulator
- **Week 5:** Analytics + Dashboard + First Benchmark

### MVP Priority System

| Priority | Scope | Value |
|----------|-------|-------|
| **P0 (Must)** | Crypto market + 3 models single agent + basic dashboard | 24/7 operable, immediate results |
| **P1 (Should)** | KRX + US markets + multi-agent + GPT-4o-mini baseline | Full benchmark complete |
| **P2 (Bonus)** | Statistical comparison + auto-reports + news sentiment | Academic completeness |

---

## Week 1 — Foundation

**Goal:** Project skeleton complete. `make install && make test` passes.

### Day 1~2: Project Setup

- `pyproject.toml`, `.env.example`, `Makefile`, `.gitignore`
- `docker-compose.yml` (PostgreSQL + TimescaleDB + Redis)
- `config/settings.py` (Pydantic Settings for env var loading)
- `config/markets.yaml`, `config/agents.yaml` drafts
- Global `CLAUDE.md` + per-module `CLAUDE.md` placement

### Day 3~4: Core Module

- `core/types.py` — all dataclass definitions (MarketSnapshot, TradingSignal, PortfolioState, etc.)
- `core/interfaces.py` — ABC definitions (BaseLLMAdapter, BaseMarketDataAdapter, etc.)
- `core/exceptions.py` — custom exception hierarchy
- `core/constants.py` — market codes, commission rates, defaults
- `tests/unit/test_types.py` — serialization/deserialization tests

### Day 5: DB Schema + Logging

- `data/db.py` — SQLAlchemy + asyncpg connection, TimescaleDB hypertable definitions
- `scripts/setup_db.py` — schema migration
- structlog configuration (JSON format, file + console output)

### Week 1 Checkpoint

```bash
make install    # Dependencies installed successfully
make db-up      # Docker containers running
make db-init    # Tables created and verified
make test-unit  # core/ tests all passing
```

---

## Week 2 — Market Data Pipeline (MIS)

**Goal:** Real-time data from 3 markets normalized into MarketSnapshot and stored in DB.

### Day 1~2: Korean Market Adapters

- `data/adapters/krx_adapter.py` — pykrx wrapper (asyncio.to_thread)
  - KOSPI200 representative stocks OHLCV, market cap, PER/PBR
  - Investor trading trends (foreign/institutional/individual)
- `data/adapters/macro_kr_adapter.py` — Bank of Korea ECOS API + OpenDART
  - Base rate, CPI, USD/KRW exchange rate
  - Financial statements via dart-fss
- Integration test: live API call -> data consistency check

### Day 3: US Market + Crypto Adapters

- `data/adapters/us_adapter.py` — EODHD + yfinance fallback
  - S&P500 major stocks OHLCV
- `data/adapters/macro_us_adapter.py` — FRED API
  - Fed Funds Rate, CPI, VIX, Treasury yields
- `data/adapters/crypto_adapter.py` — Binance WebSocket + CCXT
  - BTC/USDT, ETH/USDT, SOL/USDT real-time stream
  - Auto-reconnect + health check logic
  - Fear & Greed Index (alternative.me API)

### Day 4: Normalization + Cache

- `data/normalizer.py` — market-specific raw -> MarketSnapshot conversion
  - UTC conversion, FX integration, missing data forward-fill
- `data/cache.py` — Redis cache layer
  - Market-specific TTL (KRX: 60s, Crypto: 10s)
- `tests/unit/test_normalizer.py` — mock data conversion accuracy tests

### Day 5: Scheduler + News + E2E

- `data/scheduler.py` — APScheduler market-specific collection schedules + EventTrigger
- `data/adapters/news_adapter.py` — Google News RSS + Finnhub + CryptoPanic
- E2E test: scheduler start -> 3-market snapshot creation -> DB storage verification

### Week 2 Checkpoint

```bash
python scripts/seed_data.py           # 3-market initial data collection
make test                              # data/ tests passing
# Redis: latest snapshot queryable
# PostgreSQL: time-series data loaded
# Logs: API call success/failure recorded
```

---

## Week 3 — LLM Adapters + Single Agent

**Goal:** All models accept MarketSnapshot and return TradingSignal. Single agent benchmark possible.

### Day 1: LLM Common Infrastructure

- `llm/base.py` — BaseLLMAdapter implementation helpers (retry, timeout, logging)
- `llm/cost_tracker.py` — per-model token/cost tracking middleware
- `llm/response_parser.py` — unified parser (JSON + XML + regex fallback)
- `llm/call_logger.py` — full prompt/response recording

### Day 2: 4 Provider Adapters (parallelizable)

- `llm/deepseek_adapter.py` — OpenAI SDK compatible, reasoning_content extraction
- `llm/gemini_adapter.py` — google-genai SDK, JSON mode
- `llm/claude_adapter.py` — anthropic SDK, prompt caching, XML output
- `llm/gpt_adapter.py` — OpenAI SDK, JSON mode (reference baseline)

Completion criteria per adapter: sample MarketSnapshot input -> valid TradingSignal output

### Day 3: Prompt Engineering

- `llm/prompt_templates/single_agent.py` — single agent prompts
  - System prompt: trading rules, output format, risk guidelines
  - User prompt: MarketSnapshot summary + current portfolio
- Per-model output format optimization (DeepSeek->JSON, Gemini->JSON, Claude->XML, GPT->JSON)
- Prompt A/B test: identical snapshot -> 3-model output comparison, parse success rate check

### Day 4~5: Single Agent + Orchestrator Basics

- `agents/single_agent.py` — snapshot + portfolio -> LLM call -> signal
- `agents/orchestrator.py` — 4 single agents + buy-and-hold parallel execution
- `tests/integration/test_llm_adapters.py` — live API call tests
- `tests/integration/test_full_cycle.py` — 1-cycle E2E (data collection -> signal generation)

### Week 3 Checkpoint

```bash
# All 4 models return valid TradingSignals
# cost_tracker recording token usage
# Parse success rate > 95%
# 1-cycle E2E: MarketSnapshot -> 4 signals + buy-and-hold -> logged
```

---

## Week 4 — Multi-Agent Pipeline + Simulator

**Goal:** LangGraph multi-agent pipeline complete. Virtual trading simulator operational. All combinations runnable.

### Day 1~2: Multi-Agent Node Implementation

- `agents/multi_agent/graph.py` — LangGraph StateGraph definition
- `agents/multi_agent/analysts.py` — 4 analysts (parallel fan-out/fan-in)
  - Fundamental: financial statements, PER/PBR, earnings analysis
  - Technical: RSI, MACD, Bollinger Bands, moving averages (calculated in Python, interpreted by LLM)
  - Sentiment: social/news sentiment analysis
  - News: macro events, corporate news evaluation
- `agents/multi_agent/researchers.py` — Bull/Bear debate (2 rounds)
- Prompts: per-role files in `llm/prompt_templates/`

### Day 3: Trader + Risk + Fund Manager

- `agents/multi_agent/trader_node.py` — research synthesis -> trade proposal
- `agents/multi_agent/risk_node.py` — risk evaluation + conditional VETO edge
  - Checks: max position 30% exceeded? Cash <20%? MDD threshold? Volatility anomaly?
- `agents/multi_agent/fund_manager_node.py` — final approval/rejection
- Graph wiring: conditional edge (risk VETO -> trader re-execution, max 2)

### Day 4: Virtual Trading Simulator

- `simulator/portfolio.py` — 9 independent portfolios (8 agents + buy-and-hold)
- `simulator/order_engine.py` — slippage, commissions, cash verification
- `simulator/pnl_calculator.py` — realized/unrealized P&L, FX conversion
- `simulator/position_tracker.py` — position history DB storage
- `tests/unit/test_order_engine.py` — commission calculation, edge cases

### Day 5: Orchestrator Completion + Full Cycle Test

- `orchestrator.py` update: 8 combinations (4 models x 2 architectures) + buy-and-hold parallel
- Full cycle E2E: data collection -> snapshot -> 8 signals -> order execution -> portfolio update
- Error isolation test: force 1 agent failure -> verify rest continue normally

### Week 4 Checkpoint

```bash
# LangGraph multi-agent: 4 analysts parallel -> debate -> final signal
# 8 combinations + buy-and-hold = 9 independent portfolios simultaneous
# 1-cycle full E2E passing (data -> signal -> execution -> P&L)
# Error isolation confirmed (1 failure, rest normal)
# All trades have reasoning recorded
```

---

## Week 5 — Analytics + Dashboard + First Benchmark

**Goal:** Dashboard complete. 48~72h continuous benchmark running. First report generated.

### Day 1~2: Analytics Engine

- `analytics/metrics.py` — Sharpe, Sortino, MDD, Calmar, Win Rate, CAA, full implementation
- `analytics/comparator.py` — model comparison (t-test, Kruskal-Wallis, bootstrap CI, rolling window)
- `analytics/behavioral_profiler.py` — action distribution, conviction, contrarian score, reasoning keywords
- `analytics/audit_trail.py` — decision rationale query API
- `analytics/exporter.py` — CSV/JSON report generation
- `tests/unit/test_metrics.py` — known-data metric calculation accuracy verification

### Day 3~4: Streamlit Dashboard

- `dashboard/app.py` — main app + sidebar navigation
- 6 pages in implementation order:

| Page | Core Visualization | Priority |
|------|-------------------|----------|
| Overview | 9 equity curves (Plotly), 9 metrics cards | P0 |
| System Status | Agent status lights, error counter, next cycle countdown, cycle progress | P0 |
| Model Comparison | 4x2 heatmap (Sharpe/MDD/CR), radar chart | P1 |
| Market View | KRX/US/Crypto tabs with detailed charts + trade markers | P1 |
| Agent Detail | Decision timeline, reasoning full text, Bull/Bear debate log, Agent Personality | P2 |
| Cost Monitor | Per-model cumulative cost chart, per-signal unit cost comparison | P2 |

### Day 5: First Benchmark Launch

- Full system stability check (dry run 2~4 hours)
- Bug hotfixes from dry run
- Start 48~72h continuous benchmark (Phase 1: Crypto Only)
- Monitor via System Status page

### Week 5 Checkpoint

```bash
make dashboard   # 6 pages rendering correctly
make run         # Benchmark 48h+ continuous execution
# Overview: 9 equity curves updating in real-time
# System Status: all agents showing "healthy"
# Error rate < 5% (per-cycle basis)
# Cost Monitor: actual cost tracking functional
```

---

## Claude Code Development Strategy

### Session Management Principles

1. **Separate sessions by module.** Don't work on `data/` and `llm/` simultaneously in one session.
2. **Session start:** Read the relevant module's `CLAUDE.md` + `core/types.py` + `core/interfaces.py`.
3. **Before 50% context:** use `/compact` or `/clear`, carry forward only essential information.
4. **SCRATCHPAD.md** at project root for progress recording between sessions.

### Parallel Development Points (git worktree)

- **Week 3 Day 2:** 4 LLM adapters on 4 branches simultaneously
- **Week 4 Day 1~2:** 4 analyst nodes in parallel
- **Week 5 Day 3~4:** Dashboard pages split by priority

### Testing Strategy

- **Unit tests:** mock-based, written immediately upon module completion
- **Integration tests:** live API calls, concentrated in Weeks 2~3
- **E2E tests:** from Week 3 onward, 1 full cycle daily
- **Benchmark stability:** Week 5 dry run, minimum 4h uninterrupted confirmation
