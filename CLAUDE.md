# ATLAS — AI Trading Lab for Agent Strategy

## Project Overview

Benchmark system comparing **4 LLMs x 2 agent architectures x 10 global markets** via real-time virtual trading. Each market runs 9 independent portfolios (4 models x 2 architectures + Buy & Hold baseline). All LLM calls are cost-tracked and logged. RL-based position sizing adjusts signal weights using a DQN with experience replay.

**Scale:** ~107 Python files (19,500 LOC) · 23 TSX/TS files (2,300 LOC) · 307 tests (4,800 LOC) · 5 deploy configs

## Build & Run

```bash
pip install -e ".[dev]"          # Install with dev dependencies
make test                        # Run all 307 tests
make test-unit                   # Unit tests only
make test-integration            # Integration tests (E2E + multi-agent)
make lint                        # ruff check + format check
make format                      # ruff fix + ruff format
make typecheck                   # mypy src/
make run                         # Start benchmark (CRYPTO, unlimited)
make dashboard                   # Launch Streamlit dashboard
make db-up / db-down             # Docker compose PostgreSQL + Redis
make db-init                     # Create schema via scripts/setup_db.py
make clean                       # Remove __pycache__, .pytest_cache
```

Entry point: `python scripts/run_benchmark.py --market US CRYPTO --cycles 10`

## Tech Stack

- **Runtime**: Python 3.11+, asyncio, structlog
- **LLM Orchestration**: LangGraph (multi-agent state machine, TypedDict state)
- **Data**: yfinance, pykrx, Binance WebSocket, FRED API, feedparser
- **Storage**: PostgreSQL + TimescaleDB, Redis (cache), JSONL (results)
- **Simulation**: Custom order engine, portfolio manager, PnL calculator
- **Analytics**: Walk-forward, attribution, regime detection, calibration (numpy, scipy, scikit-learn)
- **RL**: Position sizing with PyTorch (GPU optional, CPU fallback), DQN + experience replay
- **Dashboard**: Streamlit + Plotly (9 pages, all wired)
- **Frontend**: Next.js 14 + TypeScript + Tailwind (10 pages, 8 components)
- **Reports**: Excel (openpyxl), Word (python-docx), PDF (reportlab)
- **Deploy**: Oracle Cloud, Docker Compose (5 services), nginx, Let's Encrypt SSL
- **Build**: setuptools, pyproject.toml with optional deps: `[reports]`, `[gpu]`, `[dev]`
- **Linting**: ruff (line-length 100, select E/F/I/N/UP/ANN/B/SIM)

## Absolute Rules (Fix Immediately on Violation)

1. **All LLM API calls must go through cost_tracker middleware** — no direct API calls
2. **No hardcoded API keys** — must flow through `.env` → `config/settings.py` (Pydantic Settings)
3. **All IO functions must be async/await** — no synchronous blocking calls
4. **Timestamps are always UTC** — convert to KST only for display
5. **LLM parse failure → return HOLD** — never crash the full cycle on parse error
6. **Each agent combination has independent PortfolioState** — no shared state between agents
7. **MarketSnapshot is immutable** — no modification after creation
8. **All trading signals must have reasoning** — empty string forbidden
9. **Use structlog** — no `print()`, no stdlib `logging` module
10. **100% type hints** — `Any` type forbidden
11. **All LLM calls must be recorded via call_logger** — prompt + raw response stored
12. **Multi-agent pipeline must use `call_with_prompt()`** — never `generate_signal()` for role-specific calls
13. **New module → corresponding test file in `tests/unit/`** — no untested modules

## Directory Structure

```
src/
├── core/               # Types, interfaces (ABCs), constants, exceptions, logging
│   ├── types.py        # TradingSignal, MarketSnapshot, PortfolioState, Position, Trade, enums
│   ├── interfaces.py   # BaseLLMAdapter, BaseDataAdapter, BaseAgent ABCs
│   ├── constants.py    # Market configs, initial capital, fee rates
│   ├── exceptions.py   # Custom exception hierarchy
│   └── logging.py      # structlog configuration
│
├── data/               # Market data collection & normalization
│   ├── adapters/       # 19 adapters: 10 market + 4 macro + news + onchain + social
│   │   ├── yfinance_base.py  # YFinanceBaseAdapter + YFinanceAdapterConfig (shared by 7 adapters)
│   │   ├── krx_adapter.py    # pykrx (async wrapped via to_thread)
│   │   ├── us_adapter.py     # yfinance (module-level import, v1)
│   │   ├── crypto_adapter.py # Binance WebSocket (module-level import, v1)
│   │   ├── jpx_adapter.py    # YFinanceBaseAdapter subclass
│   │   ├── sse_adapter.py    # YFinanceBaseAdapter subclass
│   │   ├── hkex_adapter.py   # YFinanceBaseAdapter subclass
│   │   ├── euronext_adapter.py # YFinanceBaseAdapter subclass
│   │   ├── lse_adapter.py    # YFinanceBaseAdapter subclass
│   │   ├── bond_adapter.py   # YFinanceBaseAdapter subclass (per_field=None)
│   │   └── commodities_adapter.py # YFinanceBaseAdapter subclass (use_info=False)
│   ├── normalizer.py   # Raw data → MarketSnapshot
│   ├── collector.py    # Orchestrates adapter calls
│   ├── cache.py        # Redis cache layer (NOT YET INTEGRATED)
│   ├── scheduler.py    # Market-hours-aware scheduling (NOT YET INTEGRATED)
│   ├── indicators.py   # Technical indicator calculations
│   └── db.py           # PostgreSQL connection pool + SQLAlchemy tables
│
├── llm/                # LLM provider layer
│   ├── base.py         # BaseLLMAdapterImpl — retry, timeout, cost tracking, call_with_prompt()
│   ├── deepseek_adapter.py   # OpenAI SDK with base_url override
│   ├── gemini_adapter.py     # Google AI SDK
│   ├── claude_adapter.py     # Anthropic SDK
│   ├── gpt_adapter.py        # OpenAI SDK
│   ├── cost_tracker.py       # Real-time USD cost metering per model
│   ├── call_logger.py        # Prompt + response persistence
│   ├── response_parser.py    # JSON/text → TradingSignal
│   ├── hallucination_detector.py  # Output validation
│   └── prompt_templates/     # Role-specific system prompts
│       ├── analyst.py        # build_fundamental/technical/sentiment/news_prompt(market)
│       ├── researcher.py     # build_bull/bear_prompt(), build_debate_rebuttal_prompt(side)
│       ├── trader.py         # build_trader_prompt()
│       ├── risk_manager.py   # build_risk_manager_prompt()
│       ├── fund_manager.py   # build_fund_manager_prompt()
│       └── single_agent.py   # Default single-agent prompt
│
├── agents/             # Agent execution layer
│   ├── single_agent.py       # snapshot → LLM → signal (1 call)
│   ├── multi_agent/graph.py  # LangGraph 5-stage pipeline (8+ calls)
│   ├── orchestrator.py       # Runs all model×arch combinations per cycle
│   ├── consensus.py          # Multi-signal aggregation (NOT YET INTEGRATED)
│   ├── reflection.py         # Self-critique loop
│   └── custom_strategy.py    # User-defined strategy support
│
├── simulator/          # Virtual trading engine
│   ├── order_engine.py       # Stateless: signal → risk check → slippage → Trade | None
│   ├── portfolio.py          # PortfolioManager: 9 portfolios/market, cash + positions
│   ├── pnl_calculator.py     # Realized + unrealized PnL
│   ├── baselines.py          # Buy & Hold, Momentum, Random, Mean-Reversion, Equal-Weight
│   ├── risk_engine.py        # Position limits, drawdown checks
│   ├── position_tracker.py   # Position lifecycle tracking
│   ├── replay.py             # Historical replay engine
│   ├── simulation_controller.py   # Lifecycle orchestration
│   └── multi_asset_optimizer.py   # Cross-asset allocation (numpy, ridge fallback)
│
├── analytics/          # Performance analysis
│   ├── metrics.py            # Sharpe, Sortino, max drawdown, win rate, cost-adjusted alpha
│   ├── results_store.py      # JSONL append-only persistence
│   ├── comparator.py         # Head-to-head model comparison
│   ├── behavioral_profiler.py # Agent behavior pattern analysis
│   ├── audit_trail.py        # Decision traceability
│   ├── exporter.py           # Data export utilities
│   ├── regime_detector.py    # Market regime classification
│   ├── walk_forward.py       # Walk-forward analysis + Monte Carlo
│   ├── calibration.py        # Prediction calibration metrics
│   └── attribution.py        # Performance attribution
│
├── rl/                 # Reinforcement learning
│   ├── position_sizer.py     # RL-based position sizing (CPU, DQN + experience replay)
│   ├── gpu_position_sizer.py # GPU-accelerated (falls back to CPU when PyTorch absent)
│   └── execution_timer.py    # Execution latency tracking
│
├── reports/            # Report generation (optional dep: [reports])
│   ├── report_factory.py     # Unified factory interface
│   ├── excel_report.py       # openpyxl
│   ├── word_report.py        # python-docx
│   └── pdf_report.py         # reportlab
│
├── saas/               # Multi-tenant support
│   ├── tenant.py             # JWT (HS256 self-impl) + API key + TenantManager (in-memory)
│   └── usage.py              # Metering + quota enforcement (in-memory)
│
└── dashboard/          # Streamlit dashboard (9 pages, all wired)
    ├── app.py
    └── data_loader.py

web/                    # Next.js 14 marketing site (10 pages, 8 components)
├── app/                # Pages: overview, model-comparison, risk, regime, simulation, strategy, optimizer, reports, system-status, replay
├── components/         # Sidebar, MetricCard, DataTable, StatusBadge, PlotlyChart, ExportButton, LanguageToggle, Providers
└── lib/                # api.ts, i18n.ts

api/                    # Vercel serverless Python (4 endpoints: health, portfolios, regime, risk)
├── lib/                # auth.py (Bearer token), db.py (MockProvider + PostgreSQLProvider)
└── *.py                # handler classes

deploy/oracle/          # Docker Compose production stack
├── docker-compose.prod.yml  # 5 services: postgres, redis, backend, nginx, certbot
├── Dockerfile               # Multi-stage Python build (ARM64 optimized)
├── nginx.conf               # Reverse proxy + SSL + rate limiting
├── init-db.sql              # 11 tables + TimescaleDB hypertables + continuous aggregates
└── setup.sh                 # Oracle Cloud setup script

.claude/skills/         # 8 Claude Code skills for development
├── fastapi-pro/        # OAuth2/JWT, async API, Pydantic V2
├── nextjs-app-router-patterns/ # Server Components, auth context, middleware
├── docker-expert/      # Multi-stage builds, Compose orchestration
├── postgres-best-practices/  # Query optimization, TimescaleDB, RLS
├── ui-ux-pro-max/      # Design system, SaaS/fintech UI rules
├── frontend-design/    # Distinctive interfaces (Anthropic official)
├── e2e-testing/        # Playwright POM, OAuth mocking
└── tailwind-design-system/ # Design tokens, component library

config/settings.py      # Pydantic Settings (reads .env)
scripts/run_benchmark.py # Main entry point (BenchmarkRunner)
tests/
├── unit/               # 294 unit tests
└── integration/        # 13 integration tests (E2E pipeline + multi-agent)
```

## Key Architecture Patterns

### LLM Adapters
- All inherit from `BaseLLMAdapterImpl` (in `src/llm/base.py`)
- Two public methods:
  - `generate_signal(snapshot, portfolio)` — uses adapter's default internal prompt
  - `call_with_prompt(system_prompt, user_prompt, snapshot, portfolio)` — uses caller-supplied prompts
- Both share the same retry/timeout/cost-tracking/call-logging logic
- Parse failure → `_fallback_hold()` → returns HOLD signal (never crashes)

### Multi-Agent Pipeline (graph.py)
- LangGraph state machine with TypedDict (NOT dataclass)
- 5 stages, each using `call_with_prompt()` with role-specific prompts:
  1. **4 Analysts** (parallel) — fundamental, technical, sentiment, news
  2. **Bull/Bear Debate** (2 rounds) — opposing arguments + rebuttals
  3. **Trader** — synthesizes debate into trade proposal
  4. **Risk Manager** — APPROVE or VETO (max 2 retries on VETO)
  5. **Fund Manager** — final signal with position sizing
- Context flows forward: each stage's output is injected into the next stage's user_prompt

### Orchestrator
- Loops over all `(model, architecture)` combinations per cycle
- SingleAgent for SINGLE, MultiAgentPipeline for MULTI
- Buy & Hold baseline runs separately with dedicated portfolio lookup
- Results persisted to JSONL via `ResultsStore`

### Data Adapters
- All inherit from `BaseDataAdapter` ABC
- 7 yfinance adapters (JPX, SSE, HKEX, EURONEXT, LSE, BOND, COMMODITIES) inherit from `YFinanceBaseAdapter` in `yfinance_base.py`
- `YFinanceAdapterConfig` frozen dataclass controls: market, currency, symbols, field extraction, info extras
- v1 adapters (US, Crypto) still use module-level yfinance/binance import
- pykrx is synchronous — always wrap with `asyncio.to_thread()`
- Macro adapters lazy-import fredapi

### Portfolio Management
- `PortfolioManager` holds 9 portfolios per market (4 models × 2 architectures + B&H)
- `get_state(model, architecture, market)` → agent portfolio
- `get_buy_hold_state(market)` → baseline portfolio
- `set_state()` writes back after trades
- B&H uses `_BUY_HOLD_MODEL = ModelProvider.DEEPSEEK`, `_BUY_HOLD_ARCH = AgentArchitecture.SINGLE` as placeholder enums

### RL Position Sizing
- `GPUPositionSizer` (falls back to `RLPositionSizer` when PyTorch absent)
- `decide(signal, portfolio)` → `SizingDecision` with `scaled_weight`
- Applied via `dataclasses.replace(sig, weight=sizing.scaled_weight)` before `execute_signal()`
- Q-learning update after each trade with normalized reward
- Model saved/loaded at cycle boundaries

### Crash Recovery
- `_try_restore()` in `BenchmarkRunner` reads `equity_curves.jsonl` + `status.json`
- Matches portfolios by `(model, architecture, market)` tuple (NOT portfolio_id, since UUIDs regenerate)
- Uses `seen` set to prevent B&H duplicate records from overwriting agent portfolios

## Common Pitfalls (Avoid These)

- `pip install uuid7` installs as `uuid_extensions` — import: `from uuid_extensions import uuid7`
- DeepSeek API uses OpenAI SDK — `openai.AsyncOpenAI(base_url=...)`
- GPT-4o-mini adapter is structurally identical to DeepSeek (both OpenAI SDK)
- Binance WebSocket disconnects — auto-reconnect logic mandatory
- Streamlit has its own event loop — use `nest_asyncio` if asyncio conflicts arise
- hatchling can fail as build backend — use setuptools instead
- `src/reports/` and `web/app/reports/` may be gitignored — use `git add -f`
- ExportButton in frontend is **default export** (NOT named export)
- Async controller tests: use `asyncio.get_event_loop().run_until_complete()`
- B&H portfolio shares `(DEEPSEEK, SINGLE)` enum keys with real agent — handle collisions in crash recovery

## Testing Conventions

- **307 total tests**: 294 unit + 13 integration
- All tests run without API keys (fully mocked)
- `pytest.ini` / `pyproject.toml`: `asyncio_mode = "auto"`
- Test files mirror source: `src/simulator/order_engine.py` → `tests/unit/test_order_engine.py`
- Mock LLM adapter pattern: subclass `BaseLLMAdapter`, return canned `TradingSignal`
- Prompt routing tests: use `PromptCapturingAdapter` to verify system prompts reach correct stages

## SaaS Upgrade Roadmap (v2.0)

### Planned Architecture
```
Browser → Nginx (SSL) →
  /api/*  → FastAPI (atlas-api container, port 8000)
  /       → Next.js (atlas-frontend container, port 3000)

FastAPI (src/api/):
  /api/auth/login/{provider}     → Google/GitHub OAuth redirect
  /api/auth/callback/{provider}  → OAuth callback → JWT cookie → redirect
  /api/auth/me                   → Current user info
  /api/auth/logout               → Clear session
  /api/portfolios                → Tenant-scoped portfolio data
  /api/simulations               → CRUD, creates Redis queue job
  /api/strategies                → CRUD custom strategies
  /api/usage                     → Usage metering stats

atlas-worker: polls Redis queue, runs BenchmarkRunner with tenant_id
```

### Key Integration Points
- `config/settings.py` — add OAuth (Google/GitHub client IDs) + JWT settings
- `src/saas/tenant.py` — replace in-memory `TenantManager` with DB-backed `TenantRepository`
- `deploy/oracle/init-db.sql` — add `provider`, `provider_id`, `avatar_url` to tenants table + simulation_queue table
- `deploy/oracle/docker-compose.prod.yml` — split into atlas-api + atlas-worker + atlas-frontend
- `web/` — add login page, auth context, route middleware, user dashboard

### Dependencies to Add
```
fastapi>=0.115, uvicorn[standard]>=0.30, python-multipart>=0.0.9,
authlib>=1.3, itsdangerous>=2.2
```

### Installed Skills
8 Claude Code skills in `.claude/skills/` for guided development:
fastapi-pro, nextjs-app-router-patterns, docker-expert, postgres-best-practices,
ui-ux-pro-max, frontend-design, e2e-testing, tailwind-design-system
