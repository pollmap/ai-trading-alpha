# ATLAS — AI Trading Lab for Agent Strategy

> **4 LLMs x 2 Agent Architectures x 10 Global Markets** — the world's first open-source benchmark for comparing AI trading agents head-to-head with real market data, virtual portfolios, and full cost accounting.

---

## Why ATLAS?

Every LLM claims to be "great at reasoning." But which one actually makes money?

ATLAS answers this by running **4 different LLMs** through **2 distinct agent architectures** across **10 global markets** simultaneously. Each combination manages its own $100K virtual portfolio. Every trade, every LLM call, every dollar of API cost is tracked. At the end, you know exactly which AI agent wins — and at what cost.

### What Makes ATLAS Different

| Feature | ATLAS | Typical AI Trading Bot |
|---------|-------|----------------------|
| LLM comparison | 4 models head-to-head | Single model |
| Agent architecture | Single + Multi-agent (5-stage debate) | Simple prompt-response |
| Markets | 10 global (stocks, crypto, bonds, commodities) | Usually 1 |
| Cost tracking | Every API call metered in USD | None |
| Baselines | Buy & Hold + 4 strategy baselines | None or basic |
| Position sizing | RL-based (DQN with experience replay) | Fixed or rule-based |
| Portfolios | 9 independent per market | Single portfolio |
| Crash recovery | Resume from last cycle via JSONL state | Start over |

---

## Execution Matrix

```
                    ┌─────────────────────────────────────┐
                    │         Per Market (x10)             │
                    ├─────────────────────────────────────┤
                    │                                      │
 4 LLM Providers    │  DeepSeek  ──┬── Single Agent (x1)  │
                    │  Gemini    ──┤   snapshot → LLM →    │
                    │  Claude    ──┤   signal (1 call)     │
                    │  GPT-4o    ──┤                       │
                    │              ├── Multi Agent (x1)    │
                    │              │   4 Analysts →        │
                    │              │   Bull/Bear Debate →  │
                    │              │   Trader → Risk Mgr → │
                    │              │   Fund Mgr (8+ calls) │
                    │              │                       │
                    │  + Buy & Hold Baseline               │
                    │                                      │
                    │  = 9 independent portfolios          │
                    └─────────────────────────────────────┘
```

**4 LLMs:** DeepSeek (OpenAI SDK) · Gemini (Google AI) · Claude (Anthropic) · GPT-4o-mini (OpenAI)

**2 Architectures:**
- **Single Agent** — 1 LLM call per cycle: snapshot → analysis → trade signal
- **Multi Agent** — 8+ LLM calls per cycle through a 5-stage LangGraph pipeline with role-specific prompts

**10 Markets:**

| Market | Data Source | Currency | Initial Capital |
|--------|-----------|----------|-----------------|
| KRX (Korea) | pykrx | KRW | ₩100,000,000 |
| US | yfinance | USD | $100,000 |
| CRYPTO | Binance WebSocket | USDT | $100,000 |
| JPX (Japan) | yfinance | JPY | ¥10,000,000 |
| SSE (China) | yfinance | CNY | ¥500,000 |
| HKEX (Hong Kong) | yfinance | HKD | HK$500,000 |
| EURONEXT (Europe) | yfinance | EUR | €100,000 |
| LSE (London) | yfinance | GBP | £100,000 |
| BOND | yfinance | USD | $100,000 |
| COMMODITIES | yfinance | USD | $100,000 |

---

## Multi-Agent Pipeline (5 Stages)

The Multi-Agent architecture uses a LangGraph state machine where each stage calls the LLM with a specialized system prompt. Context flows forward — each stage's output feeds into the next.

```
Stage 1: 4 Analysts (parallel)
  ├── Fundamental Analyst   →  earnings, valuation, macro outlook
  ├── Technical Analyst     →  patterns, indicators, support/resistance
  ├── Sentiment Analyst     →  social media, fear/greed, positioning
  └── News Analyst          →  headlines, events, regulatory changes
                ▼ analyst reports
Stage 2: Bull/Bear Debate (2 rounds)
  ├── Bull Researcher       →  arguments FOR buying
  ├── Bear Researcher       →  arguments AGAINST buying
  └── Rebuttals             →  counterarguments from each side
                ▼ debate transcript
Stage 3: Trader             →  synthesizes into trade proposal (symbol, action, weight)
                ▼ trade proposal
Stage 4: Risk Manager       →  APPROVE or VETO (up to 2 retries on VETO)
                ▼ risk assessment
Stage 5: Fund Manager       →  final signal with position sizing and reasoning
```

---

## Features

### Core Engine
- **Order Engine** — stateless trade execution: signal → risk check → slippage → commission → `Trade | None`
- **Portfolio Manager** — 9 isolated portfolios per market, cash tracking, position lifecycle
- **PnL Calculator** — realized + unrealized P&L per portfolio
- **Risk Engine** — position limits, drawdown circuit breaker, daily loss limits, confidence thresholds
- **RL Position Sizer** — DQN-based sizing with experience replay and epsilon-greedy exploration (GPU optional)
- **Crash Recovery** — resume benchmark from JSONL state after process restart

### Data Layer
- **19 adapters**: 10 market + 4 macro (FRED) + news + on-chain + social sentiment
- **7 yfinance adapters** share `YFinanceBaseAdapter` with config-driven field extraction
- **Technical Indicators** — SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- **Redis cache** layer for market data
- **Market-hours scheduler** — respects exchange trading hours per market

### LLM Layer
- **4 provider adapters** — all inherit from `BaseLLMAdapterImpl` with shared retry/timeout/cost logic
- **Cost Tracker** — real-time USD cost per model per call
- **Call Logger** — every prompt + raw response persisted to JSONL
- **Response Parser** — JSON/text → `TradingSignal` with fallback to HOLD on parse failure
- **Hallucination Detector** — validates LLM outputs against market constraints
- **6 prompt templates** — role-specific system prompts for each pipeline stage

### Analytics
- **Metrics** — Sharpe, Sortino, max drawdown, win rate, cost-adjusted alpha
- **Walk-Forward Analysis** — rolling window backtesting + Monte Carlo simulation
- **Regime Detection** — bull/bear/sideways classification per market
- **Performance Attribution** — decompose returns by factor
- **Calibration** — measure prediction confidence accuracy
- **Behavioral Profiler** — analyze agent trading patterns
- **Audit Trail** — full decision traceability from signal to trade

### Dashboards & Reports
- **Streamlit Dashboard** (9 pages) — real-time equity curves, trade log, cost analysis, risk metrics
- **Next.js Frontend** (10 pages) — portfolio overview, model comparison, regime analysis, strategy builder
- **Report Factory** — export to Excel (openpyxl), Word (python-docx), PDF (reportlab)

### SaaS Infrastructure
- **Multi-tenant** — JWT + API key authentication, 4 plan tiers (Free/Starter/Pro/Enterprise)
- **Usage Metering** — quota enforcement per tenant (API calls, simulations, strategies)
- **PostgreSQL + TimescaleDB** — 11-table schema with hypertables and continuous aggregates
- **Docker Compose** — 5-service production stack (postgres, redis, backend, nginx, certbot)

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd ai-trading-alpha
pip install -e ".[dev]"

# 2. Configure API keys
cp .env.example .env
# Edit: DEEPSEEK_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY

# 3. Run benchmark
python scripts/run_benchmark.py --market CRYPTO --cycles 5

# 4. View results
streamlit run src/dashboard/app.py
```

### Benchmark Commands

```bash
# Single market, 10 cycles
python scripts/run_benchmark.py --market US --cycles 10

# Multiple markets, 5 cycles each
python scripts/run_benchmark.py --market US CRYPTO KRX --cycles 5

# All 10 markets, unlimited cycles
python scripts/run_benchmark.py --market KRX US CRYPTO JPX SSE HKEX EURONEXT LSE BOND COMMODITIES
```

### Makefile

| Command | Description |
|---------|-------------|
| `make test` | Run all 307 tests (unit + integration) |
| `make test-unit` | Unit tests only (294) |
| `make test-integration` | Integration tests only (13) |
| `make lint` | ruff check + format check |
| `make format` | ruff fix + ruff format |
| `make typecheck` | mypy src/ |
| `make run` | Start benchmark (CRYPTO, unlimited) |
| `make dashboard` | Launch Streamlit dashboard |
| `make db-up` / `make db-down` | Docker Compose PostgreSQL + Redis |
| `make db-init` | Create database schema |
| `make clean` | Remove `__pycache__`, `.pytest_cache` |

---

## Architecture

### CUFA Integration Blueprint

If you want to integrate ATLAS with **CUFA NEXUS / CUFA Web**, start from the staged guide below:

- `docs/cufa-integration.md`

The document covers:
- target architecture split (ATLAS engine API vs NEXUS orchestration vs Web UI),
- auth/job-sync/billing risks and mitigations,
- phased rollout plan,
- local/prod deployment playbooks.

```
┌─────────────────────────────────────────────────────────────────┐
│                         ATLAS Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │  19 Data      │───▶│  Collector   │───▶│  MarketSnapshot  │    │
│  │  Adapters     │    │  + Normalizer│    │  (immutable)     │    │
│  │  yfinance,    │    │  + Indicators│    │                  │    │
│  │  pykrx,       │    │              │    │                  │    │
│  │  Binance WS,  │    │              │    │                  │    │
│  │  FRED, news   │    │              │    │                  │    │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘    │
│                                                    │              │
│                          ┌─────────────────────────┤              │
│                          ▼                         ▼              │
│  ┌──────────────────────────────┐  ┌───────────────────────┐    │
│  │     Orchestrator              │  │  Baselines             │    │
│  │                               │  │  Buy & Hold            │    │
│  │  ┌─────────┐  ┌────────────┐ │  │  Momentum              │    │
│  │  │ Single  │  │ Multi-Agent│ │  │  Random                 │    │
│  │  │ Agent   │  │ Pipeline   │ │  │  Mean-Reversion         │    │
│  │  │ (1 call)│  │ (8+ calls) │ │  │  Equal-Weight           │    │
│  │  │         │  │            │ │  └──────────┬────────────┘    │
│  │  │ x4 LLMs │  │ x4 LLMs   │ │             │                  │
│  │  └────┬────┘  └─────┬─────┘ │             │                  │
│  │       └──────┬───────┘       │             │                  │
│  └──────────────┼───────────────┘             │                  │
│                  ▼                             ▼                  │
│          TradingSignal[]                TradingSignal[]          │
│                  │                             │                  │
│                  └─────────────┬───────────────┘                  │
│                                ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │            RL Position Sizer (DQN)                         │   │
│  │  signal.weight adjusted by learned sizing policy           │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                              ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              Order Engine (stateless)                       │   │
│  │  signal → risk check → slippage → commission → Trade|None  │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                              ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │           Portfolio Manager (9 per market)                  │   │
│  │  cash tracking · position updates · PnL calculation         │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                              ▼                                    │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  Results Store (JSONL + PostgreSQL)                         │   │
│  │  equity_curves · signals · trades · costs · snapshots       │   │
│  └──────────────────────────┬────────────────────────────────┘   │
│                              ▼                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │ Streamlit (9pg)│  │ Next.js (10pg) │  │ Reports (PDF/    │   │
│  │ Analytics      │  │ Dashboard UI   │  │ Excel/Word)      │   │
│  └────────────────┘  └────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ai-trading-alpha/
├── config/                      # Pydantic Settings, markets.yaml
│   └── settings.py              # All env vars flow through here
├── scripts/
│   └── run_benchmark.py         # Main entry point (BenchmarkRunner)
├── src/
│   ├── core/                    # Types, interfaces (ABCs), constants, exceptions, logging
│   ├── data/                    # 19 adapters + normalizer + cache + scheduler + indicators
│   │   └── adapters/            # YFinanceBaseAdapter (shared) + 10 market adapters
│   ├── llm/                     # 4 LLM adapters + cost tracker + call logger + parser
│   │   └── prompt_templates/    # 6 role-specific system prompts
│   ├── agents/                  # SingleAgent, MultiAgentPipeline (LangGraph), orchestrator
│   ├── simulator/               # OrderEngine, PortfolioManager, PnL, baselines, risk engine
│   ├── analytics/               # Metrics, walk-forward, attribution, regime, calibration
│   ├── rl/                      # DQN position sizer (GPU optional, CPU fallback)
│   ├── reports/                 # Excel, Word, PDF report generation
│   ├── saas/                    # Multi-tenant JWT auth + usage metering
│   └── dashboard/               # Streamlit app (9 pages)
├── web/                         # Next.js 14 + TypeScript + Tailwind (10 pages, 8 components)
├── deploy/oracle/               # Docker Compose, nginx, SSL, DB schema
├── .claude/skills/              # 8 Claude Code skills for development
├── tests/
│   ├── unit/                    # 294 unit tests
│   └── integration/             # 13 integration tests
└── data/results/                # JSONL output (equity curves, signals, trades, costs)
```

**Scale:** ~107 Python files (19,500 LOC) · 23 TSX/TS files (2,300 LOC) · 307 tests (4,800 LOC) · 5 deploy configs

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Runtime** | Python 3.11+, asyncio, structlog |
| **LLM Orchestration** | LangGraph (multi-agent state machine, TypedDict) |
| **Data** | yfinance, pykrx, Binance WebSocket, FRED API, feedparser |
| **Storage** | PostgreSQL + TimescaleDB, Redis, JSONL |
| **Simulation** | Custom order engine, portfolio manager, PnL calculator |
| **Analytics** | numpy, scipy, scikit-learn, pandas |
| **RL** | PyTorch (GPU optional, CPU fallback) |
| **Dashboard** | Streamlit + Plotly (9 pages) |
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Plotly.js |
| **Reports** | openpyxl (Excel), python-docx (Word), reportlab (PDF) |
| **Deploy** | Oracle Cloud, Docker Compose, nginx, Let's Encrypt SSL |
| **Build** | setuptools, pyproject.toml (`[reports]`, `[gpu]`, `[dev]`) |
| **Lint** | ruff (line-length 100, select E/F/I/N/UP/ANN/B/SIM) |

---

## Key Design Decisions

1. **Immutable MarketSnapshot** — created once by `DataNormalizer`, never modified. All agents receive identical data.
2. **Stateless OrderEngine** — `execute_signal(signal, portfolio, config, price) → Trade | None`. Pure function, no side effects.
3. **Independent portfolios** — each `(model, architecture, market)` combination has its own `PortfolioState`. Zero shared state between agents.
4. **Error isolation** — one agent failure → HOLD fallback. Other agents continue unaffected.
5. **Cost transparency** — every LLM call metered via `CostTracker`. Real-time USD cost per model visible in dashboard.
6. **JSONL persistence** — append-only results store. Dashboard reads live data, no ETL pipeline needed.
7. **RL integration** — `GPUPositionSizer.decide()` adjusts signal weight before order execution. Model saved/loaded across restarts.
8. **Crash recovery** — `_try_restore()` matches portfolios by `(model, arch, market)` tuple to resume after process restart.
9. **yfinance deduplication** — 7 adapters inherit from `YFinanceBaseAdapter` with `YFinanceAdapterConfig` controlling per-market behavior.

---

## Testing

**307 tests** — no API keys required, all fully mocked.

```bash
pytest tests/ -v                    # All 307 tests
pytest tests/unit/ -v               # 294 unit tests
pytest tests/integration/ -v        # 13 integration tests
pytest tests/unit/test_order_engine.py -v  # Specific module
```

| Test Suite | Count | Covers |
|------------|-------|--------|
| Core types & interfaces | 67 | Types, enums, constants, portfolio state |
| Analytics & metrics | 104 | Sharpe, walk-forward, attribution, calibration |
| Simulator | 95 | OrderEngine, PnL, baselines, risk engine, optimizer |
| yfinance base adapter | 17 | YAML loading, batch fetch, retry, 7 subclass instantiation |
| LLM adapter base | 11 | Retry, timeout, cost tracking, `call_with_prompt()` |
| E2E pipeline | 7 | Full orchestrator loop with mock LLM |
| Multi-agent routing | 6 | Prompt routing to all 5 pipeline stages |

---

## Deployment

### Local Development
```bash
pip install -e ".[dev]"
python scripts/run_benchmark.py --market CRYPTO --cycles 5
```

### Docker (Production)
```bash
cd deploy/oracle
cp ../../.env.example .env
# Edit .env with production values

docker compose -f docker-compose.prod.yml up -d
```

**Production stack:** PostgreSQL+TimescaleDB · Redis · Backend · Nginx (SSL) · Certbot (auto-renew)

### Oracle Cloud (ARM64)
```bash
# Use the provided setup script
bash deploy/oracle/setup.sh
```

Optimized for Oracle Cloud's free ARM A1 instances (4 OCPU, 24GB RAM).

---

## Dashboard Pages

### Streamlit (Analytics, 9 pages)

| Page | Description |
|------|-------------|
| Overview | Portfolio returns, active agents, cycle status |
| Equity Curves | Line charts comparing all 9 portfolios |
| Trade History | Signal and trade log with filters |
| Cost Analysis | LLM cost breakdown by model |
| Market Data | Live market snapshots and prices |
| Risk Metrics | Drawdown, Sharpe, volatility |
| Agent Comparison | Head-to-head model performance |
| Agent Detail | Individual agent deep-dive |
| Cost Monitor | Real-time API spend tracking |

### Next.js (Frontend, 10 pages)

| Page | Description |
|------|-------------|
| Overview | Portfolio summary across markets |
| Model Comparison | LLM performance matrix |
| Risk Dashboard | VaR, drawdown analysis |
| Regime Analysis | Bull/bear/sideways per market |
| Simulation | Create and monitor simulations |
| Custom Strategy | Build user-defined strategies |
| Optimizer | Multi-asset portfolio optimization |
| Reports | Generate Excel/Word/PDF |
| System Status | LLM provider health, infra |
| Replay | Historical session replay |

---

## Roadmap

### Current (v1.0)
- [x] 4 LLM providers x 2 architectures x 10 markets
- [x] Multi-agent pipeline (LangGraph, 5 stages)
- [x] RL position sizing (DQN)
- [x] Crash recovery
- [x] Streamlit dashboard (9 pages)
- [x] Next.js frontend (10 pages)
- [x] Report generation (Excel/Word/PDF)
- [x] 307 tests passing

### Next (v2.0 — SaaS Upgrade)
- [ ] FastAPI backend with REST API
- [ ] OAuth authentication (Google + GitHub)
- [ ] Multi-tenant data isolation
- [ ] Per-user simulations and custom strategies
- [ ] Redis job queue + background worker
- [ ] Docker Compose split (API + Worker + Frontend)
- [ ] PostgreSQL data persistence (replace JSONL)

---

## License

MIT
