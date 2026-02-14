# ATLAS — AI Trading Lab for Agent Strategy

> Benchmark system comparing **4 LLMs × 2 agent architectures × 10 global markets** via real-time virtual trading.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ATLAS Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │  Market   │───▶│  Data        │───▶│  MarketSnapshot  │    │
│  │  Adapters │    │  Collector   │    │  (immutable)     │    │
│  │  (17)     │    │  + Normalizer│    │                  │    │
│  └──────────┘    └──────────────┘    └────────┬─────────┘    │
│                                               │              │
│                         ┌─────────────────────┤              │
│                         ▼                     ▼              │
│  ┌─────────────────────────┐  ┌────────────────────────┐    │
│  │   Orchestrator          │  │  Buy & Hold Baseline   │    │
│  │                         │  │  (+ Momentum, Random,  │    │
│  │  ┌───────┐  ┌────────┐ │  │   Mean-Reversion,      │    │
│  │  │Single │  │ Multi  │ │  │   Equal-Weight)        │    │
│  │  │Agent  │  │ Agent  │ │  └───────────┬────────────┘    │
│  │  │       │  │5-stage │ │              │                  │
│  │  │ GPT   │  │pipeline│ │              │                  │
│  │  │Claude │  │        │ │              │                  │
│  │  │Gemini │  │Analysts│ │              │                  │
│  │  │DeepSk │  │Debate  │ │              │                  │
│  │  └───┬───┘  │Trader  │ │              │                  │
│  │      │      │Risk Mgr│ │              │                  │
│  │      │      │Fund Mgr│ │              │                  │
│  │      │      └───┬────┘ │              │                  │
│  │      └─────┬────┘      │              │                  │
│  └────────────┼───────────┘              │                  │
│               ▼                          ▼                  │
│       TradingSignal[]              TradingSignal[]          │
│               │                          │                  │
│               └──────────┬───────────────┘                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Order Engine                         │       │
│  │  signal → risk check → slippage → commission →   │       │
│  │  position limit → Trade | None                    │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │           Portfolio Manager                       │       │
│  │  9 portfolios/market (4 models × 2 arch + B&H)   │       │
│  │  cash tracking, position updates, PnL calc        │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │            Results Store (JSONL)                   │       │
│  │  equity_curves / signals / trades / costs          │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ▼                                   │
│  ┌─────────────────┐  ┌────────────────────┐                │
│  │ Streamlit (9pg) │  │ Next.js 14 (10pg)  │                │
│  │ Real-time data  │  │ Marketing site     │                │
│  └─────────────────┘  └────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd ai-trading-alpha
pip install -e ".[dev]"

# 2. Set API keys in .env
cp .env.example .env
# Edit: DEEPSEEK_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY

# 3. Run benchmark
python scripts/run_benchmark.py --market CRYPTO --cycles 5

# 4. View results on dashboard
streamlit run src/dashboard/app.py
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make test` | Run all pytest (290 tests: unit + integration) |
| `make test-unit` | Unit tests only |
| `make run` | Start benchmark (`CRYPTO`, unlimited cycles) |
| `make dashboard` | Launch Streamlit dashboard |
| `make db-up` | Start PostgreSQL + Redis (docker-compose) |
| `make db-init` | Create database schema |

## Benchmark Configuration

```bash
# Single market
python scripts/run_benchmark.py --market US --cycles 10

# Multiple markets
python scripts/run_benchmark.py --market US CRYPTO KRX --cycles 5

# All 10 global markets, unlimited
python scripts/run_benchmark.py --market KRX US CRYPTO JPX SSE HKEX EURONEXT LSE BOND COMMODITIES
```

## Execution Matrix

**4 LLM Providers:**
- DeepSeek (via OpenAI SDK, `base_url` override)
- Gemini (Google AI SDK)
- Claude (Anthropic SDK)
- GPT-4o-mini (OpenAI SDK)

**2 Agent Architectures:**

| Architecture | Pipeline | LLM Calls/Cycle |
|---|---|---|
| **Single** | `snapshot → LLM → signal` | 1 |
| **Multi** | `4 Analysts → Bull/Bear Debate → Trader → Risk Mgr → Fund Mgr` (each with role-specific prompt) | 8+ |

**+ Buy & Hold Baseline** — equal-weight initial buy, then hold forever.

**= 9 portfolios per market**, each with independent state.

## 10 Global Markets

| Market | Adapter | Currency | Initial Capital |
|---|---|---|---|
| KRX | pykrx (async wrapped) | KRW | ₩100,000,000 |
| US | yfinance | USD | $100,000 |
| CRYPTO | Binance WebSocket | USDT | $100,000 |
| JPX | yfinance | JPY | ¥10,000,000 |
| SSE | yfinance | CNY | ¥500,000 |
| HKEX | yfinance | HKD | HK$500,000 |
| EURONEXT | yfinance | EUR | €100,000 |
| LSE | yfinance | GBP | £100,000 |
| BOND | yfinance | USD | $100,000 |
| COMMODITIES | yfinance | USD | $100,000 |

## Tech Stack

- **Backend**: Python 3.11+, asyncio, structlog
- **LLM Orchestration**: LangGraph (multi-agent state machine)
- **Data**: yfinance, pykrx, Binance WebSocket, FRED API
- **Simulation**: Custom order engine, portfolio manager, PnL calculator
- **Analytics**: Walk-forward, attribution, regime detection, calibration
- **RL**: Position sizing with PyTorch (GPU optional, CPU fallback)
- **Dashboard**: Streamlit + Plotly (9 pages, live data)
- **Frontend**: Next.js 14 + TypeScript + Tailwind (marketing site)
- **Reports**: Excel, Word, PDF export via report factory
- **Deploy**: Oracle Cloud, Docker, nginx, SSL

## Project Structure

```
ai-trading-alpha/
├── config/                  # Settings (Pydantic), markets.yaml
├── scripts/
│   └── run_benchmark.py     # Main entry point
├── src/
│   ├── core/                # Types, interfaces, constants, exceptions, logging
│   ├── data/                # 17 market adapters + normalizer + cache + scheduler
│   │   └── adapters/        # KRX, US, Crypto, JPX, SSE, HKEX, EURONEXT, LSE, Bond, Commodities
│   ├── llm/                 # 4 LLM adapters + cost tracker + call logger + response parser
│   │   └── prompt_templates/# System prompts for each agent role
│   ├── agents/              # SingleAgent, MultiAgentPipeline, orchestrator
│   │   └── multi_agent/     # LangGraph 5-stage pipeline
│   ├── simulator/           # OrderEngine, PortfolioManager, PnL, baselines, risk engine
│   ├── analytics/           # Metrics, results store, comparator, profiler, audit trail
│   ├── rl/                  # RL position sizer (GPU/CPU), execution timer
│   ├── reports/             # Excel, Word, PDF report generation
│   ├── saas/                # Tenant management (JWT + API key), usage metering
│   └── dashboard/           # Streamlit app (9 pages)
├── web/                     # Next.js 14 marketing site (10 pages)
├── deploy/oracle/           # Docker, nginx, SSL, DB schema
├── tests/
│   ├── unit/                # 277 unit tests (types, metrics, simulator, LLM base)
│   └── integration/         # 13 integration tests (E2E pipeline, multi-agent routing)
└── data/results/            # JSONL output (equity curves, signals, trades, costs)
```

## Key Design Decisions

1. **Immutable MarketSnapshot** — created once by `DataNormalizer`, never modified. All agents receive the same data.
2. **Stateless OrderEngine** — `execute_signal(signal, portfolio, config, price) → Trade | None`. No internal state, pure function.
3. **Independent portfolios** — each model×architecture combination has its own `PortfolioState`. No shared state between agents.
4. **Error isolation** — one agent failure → HOLD fallback. Other agents continue unaffected.
5. **Cost tracking** — every LLM call metered via `CostTracker` context manager. Real-time USD cost per model.
6. **JSONL persistence** — append-only results store. Dashboard reads live data, no separate ETL needed.

## Multi-Agent Pipeline (5 Stages)

Each stage calls the LLM via `call_with_prompt()` with role-specific system prompts
from `src/llm/prompt_templates/`. This ensures every agent receives specialized instructions
rather than a generic default prompt.

```
Stage 1: 4 Analysts (parallel) — each with specialized system prompt
  ├── Fundamental Analyst  → build_fundamental_prompt(market)
  ├── Technical Analyst    → build_technical_prompt(market)
  ├── Sentiment Analyst    → build_sentiment_prompt(market)
  └── News Analyst         → build_news_prompt(market)
          │ analyst reports passed as context ▼
Stage 2: Bull/Bear Debate (2 rounds)
  ├── Bull Researcher → build_bull_prompt()    — arguments FOR buying
  ├── Bear Researcher → build_bear_prompt()    — arguments AGAINST
  └── Rebuttals       → build_debate_rebuttal_prompt(side)
          │ debate log passed as context ▼
Stage 3: Trader → build_trader_prompt() — synthesizes into trade proposal
          │ trade proposal passed as context ▼
Stage 4: Risk Manager → build_risk_manager_prompt() — APPROVE or VETO
          │ risk assessment passed as context ▼
Stage 5: Fund Manager → build_fund_manager_prompt() — final signal + sizing
```

## Dashboard Pages (Streamlit)

| Page | Description |
|---|---|
| Overview | Portfolio returns, active agents, cycle status |
| Equity Curves | Line charts comparing all 9 portfolios per market |
| Trade History | Signal and trade log with filters |
| Cost Analysis | LLM cost breakdown by model, cumulative spend |
| Market Data | Live market snapshots and symbol data |
| Risk Metrics | Drawdown, Sharpe ratio, volatility |
| Agent Comparison | Head-to-head model performance matrix |
| Reports | Export to Excel/Word/PDF |
| Settings | Configuration and API key status |

## Testing

**290 tests** — no API keys required, all mocked.

```bash
# All tests (290 passed)
pytest tests/ -v

# Unit tests only (277 tests)
pytest tests/unit/ -v

# Integration tests — E2E pipeline + multi-agent prompt routing (13 tests)
pytest tests/integration/ -v

# Specific module
pytest tests/unit/test_order_engine.py -v
```

| Test Suite | Count | Covers |
|---|---|---|
| Core types & interfaces | 67 | Types, enums, constants, portfolio state |
| Analytics & metrics | 104 | Metrics, walk-forward, attribution, calibration |
| Simulator | 95 | OrderEngine, PnL, baselines, risk engine, optimizer |
| LLM adapter base | 11 | Retry, timeout, cost tracking, `call_with_prompt()` |
| E2E pipeline | 7 | Full orchestrator loop with mock LLM |
| Multi-agent routing | 6 | Prompt routing to all 5 pipeline stages |

## License

MIT
