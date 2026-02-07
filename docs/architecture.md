# ATLAS - System Architecture & Module Specifications

## Project Name

**ATLAS** — AI Trading Lab for Agent Strategy

---

## Core Design Principles

### Principle 1: Data Isolation for Fair Comparison

3 core models (DeepSeek, Gemini, Claude) + 1 reference baseline (GPT-4o-mini) must receive identical data at identical timestamps. The Market Intelligence Stream (MIS) normalizes data from a single source and delivers identical payloads to each agent. Models are **prohibited** from independently fetching external data.

### Principle 2: Architecture x Model 2-Axis Evaluation

Not just "which model performs best," but measuring the **cross-effect** of agent structure (Single vs Multi) and model selection.

**Evaluation Matrix:**

|                    | DeepSeek-R1 | Gemini 2.5 Pro | Claude Sonnet 4.5 | GPT-4o-mini (ref) |
|--------------------|-------------|----------------|--------------------|--------------------|
| **Single Agent**   | Cell A      | Cell B         | Cell C             | Cell D             |
| **Multi-Agent**    | Cell E      | Cell F         | Cell G             | Cell H             |

- Each cell = per-market (KRX, US, Crypto) evaluation
- Passive baseline: Buy & Hold (always included)
- Total: 8 active portfolios + 1 Buy&Hold = **9 independent portfolios per market**

### Principle 3: Forward-Test Only (No Backtesting)

No backtesting. This eliminates data contamination concerns entirely. All trading signals are based on real-time or same-day data.

**Evaluation Phases:**
- **Phase 1 (Validation):** 72h Crypto Only — system stability check
- **Phase 2 (Main Benchmark):** Minimum 2 weeks, all markets active
  - Weekday 5 days x 2 weeks = 10 KRX/US trading days
  - Crypto 14 days continuous = ~1,344 cycles (15-min interval)
- **Phase 3 (Extended, optional):** 1 month+ long-term operation

### Principle 4: Module Independence

Each component (data collection, LLM adapter, agent pipeline, simulator, dashboard) must be independently testable and replaceable. Interfaces are defined via abstract base classes.

### Principle 5: Full Reproducibility

Every cycle's I/O is fully recorded:
1. **Input recording:** Full MarketSnapshot as JSON in DB
2. **Prompt recording:** Exact prompt text sent to each LLM in DB
3. **Response recording:** Raw LLM response in full in DB
4. **Seed fixed:** temperature=0.0 for all models

---

## High-Level Architecture

```
+-------------------------------------------------------------+
|                      ATLAS System                            |
|                                                              |
|  +-------------------------------------------------------+  |
|  |        Market Intelligence Stream (MIS)                |  |
|  |                                                        |  |
|  |  +----------+ +----------+ +-----------+               |  |
|  |  |KRX Data  | |US Data   | |Crypto     |               |  |
|  |  |Adapter   | |Adapter   | |Adapter    |               |  |
|  |  +----+-----+ +----+-----+ +----+------+               |  |
|  |       +-----------++-----------+                        |  |
|  |              +-----v-----+                              |  |
|  |              | Normalize | -> Redis Cache                |  |
|  |              | & Store   | -> PostgreSQL/TimescaleDB     |  |
|  |              +-----+-----+                              |  |
|  +--------------------+----------------------------------+  |
|                       |                                      |
|                       v Standardized MarketSnapshot           |
|                                                              |
|  +-------------------------------------------------------+  |
|  |        Agent Execution Protocol (AEP)                  |  |
|  |                                                        |  |
|  |  +--------------------------------------------------+  |  |
|  |  |          Benchmark Orchestrator                   |  |  |
|  |  |                                                   |  |  |
|  |  |  +----------+ +----------+ +----------+ +------+  |  |  |
|  |  |  |DeepSeek  | |Gemini    | |Claude    | |GPT   |  |  |  |
|  |  |  |Runner    | |Runner    | |Runner    | |Runner|  |  |  |
|  |  |  |          | |          | |          | |(ref) |  |  |  |
|  |  |  |+Single-+ | |+Single-+ | |+Single-+ | |+Sgl-+|  |  |  |
|  |  |  ||Agent  | | ||Agent  | | ||Agent  | | ||Agt ||  |  |  |
|  |  |  |+-------+ | |+-------+ | |+-------+ | |+----+|  |  |  |
|  |  |  |+Multi--+ | |+Multi--+ | |+Multi--+ | |+Mlt-+|  |  |  |
|  |  |  ||Agent  | | ||Agent  | | ||Agent  | | ||Agt ||  |  |  |
|  |  |  ||Pipe   | | ||Pipe   | | ||Pipe   | | ||Pipe||  |  |  |
|  |  |  |+-------+ | |+-------+ | |+-------+ | |+----+|  |  |  |
|  |  |  +----------+ +----------+ +----------+ +------+  |  |  |
|  |  +--------------------------------------------------+  |  |
|  +-------------------------------------------------------+  |
|                       |                                      |
|                       v TradingSignal + Reasoning             |
|                                                              |
|  +-------------------------------------------------------+  |
|  |         Virtual Trading Simulator                      |  |
|  |                                                        |  |
|  |  Portfolio Manager -> Order Execution Engine            |  |
|  |  -> P&L Calculator -> Position Tracker                  |  |
|  +-------------------------------------------------------+  |
|                       |                                      |
|                       v PortfolioState + TradeHistory          |
|                                                              |
|  +-------------------------------------------------------+  |
|  |     Performance Analytics Interface (PAI)              |  |
|  |                                                        |  |
|  |  Streamlit Dashboard -> Metrics Engine                  |  |
|  |  -> Decision Audit Trail -> Export/Report               |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

---

## Core Data Flow

Every trading cycle (15min~30min interval, configurable):

1. **MIS** collects latest data from each market -> normalizes -> creates `MarketSnapshot` object
2. `MarketSnapshot` is delivered to the **Benchmark Orchestrator**
3. Orchestrator simultaneously delivers identical snapshot to all agent combinations (4 models x 2 architectures)
4. Each agent returns a `TradingSignal` (BUY/SELL/HOLD + weight + reasoning)
5. **Virtual Trading Simulator** executes signals, updates portfolios
6. **PAI** calculates real-time metrics and refreshes dashboard

### Event-Driven Triggers (Supplement to Fixed Cycles)

In addition to regular fixed-interval cycles, emergency cycles are triggered when:
- **Price spike:** >=3% change from last snapshot
- **VIX surge:** >=20% increase from previous day
- **FX spike:** USD/KRW >=1% change
- **Extreme sentiment:** Crypto Fear&Greed <20 or >80

Cooldown: 15 minutes after trigger to prevent excessive API calls.

---

## Market Operating Schedule (KST)

| Time (KST)      | Active Markets     | Agent Behavior                       |
|------------------|--------------------|--------------------------------------|
| 00:00~06:30      | Crypto Only        | Crypto trading cycles                |
| 09:00~15:30      | KRX + Crypto       | Korean stocks + crypto in parallel   |
| 15:30~23:30      | Crypto Only        | Crypto trading cycles                |
| 23:30~06:00(+1)  | US + Crypto        | US stocks + crypto in parallel       |

---

## Core Interfaces (Abstract Classes)

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal
from abc import ABC, abstractmethod


class Market(str, Enum):
    KRX = "KRX"
    US = "US"
    CRYPTO = "CRYPTO"

class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class ModelProvider(str, Enum):
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    CLAUDE = "claude"
    GPT = "gpt"  # reference baseline

class AgentArchitecture(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


@dataclass
class SymbolData:
    symbol: str
    market: Market
    open: float
    high: float
    low: float
    close: float
    volume: float
    currency: str                      # "KRW", "USD", "USDT"
    per: float | None = None           # PER (stocks only)
    pbr: float | None = None           # PBR (stocks only)
    market_cap: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class MacroData:
    kr_base_rate: float | None = None
    us_fed_rate: float | None = None
    usdkrw: float | None = None
    vix: float | None = None
    kr_cpi_yoy: float | None = None
    us_cpi_yoy: float | None = None
    fear_greed_index: float | None = None


@dataclass
class NewsItem:
    timestamp: datetime
    title: str
    summary: str
    source: str
    relevance_score: float             # 0~1
    sentiment: float                   # -1~+1


@dataclass
class MarketSnapshot:
    """Standardized payload: MIS -> AEP"""
    snapshot_id: str                   # UUID7
    timestamp: datetime                # UTC
    market: Market
    symbols: dict[str, SymbolData]
    macro: MacroData
    news: list[NewsItem]
    metadata: dict = field(default_factory=dict)


@dataclass
class TradingSignal:
    """Trading signal returned by each agent"""
    signal_id: str                     # UUID
    snapshot_id: str                   # links back to source snapshot
    timestamp: datetime
    symbol: str
    market: Market
    action: Action
    weight: float                      # 0~1 (portfolio weight)
    confidence: float                  # 0~1
    reasoning: str                     # full decision rationale
    model: ModelProvider
    architecture: AgentArchitecture
    latency_ms: float
    token_usage: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class PortfolioState:
    """Independent portfolio per agent combination"""
    portfolio_id: str
    model: ModelProvider
    architecture: AgentArchitecture
    market: Market
    cash: float
    positions: dict[str, Position]
    initial_capital: float
    created_at: datetime


class BaseLLMAdapter(ABC):
    @abstractmethod
    async def generate_signal(
        self, snapshot: MarketSnapshot, portfolio: PortfolioState
    ) -> TradingSignal: ...

class BaseMarketDataAdapter(ABC):
    @abstractmethod
    async def fetch_latest(self) -> dict[str, SymbolData]: ...

class BaseMetricsCalculator(ABC):
    @abstractmethod
    async def calculate(self, portfolio_history: list) -> dict: ...
```

---

## Directory Structure

```
atlas/
├── CLAUDE.md                          # Global Claude Code rules
├── pyproject.toml                     # Dependencies & build config
├── .env.example                       # API key template
├── docker-compose.yml                 # PostgreSQL + Redis + TimescaleDB
├── Makefile                           # Common command shortcuts
│
├── config/
│   ├── settings.py                    # Pydantic Settings (env var loader)
│   ├── markets.yaml                   # Per-market config (symbols, schedule, thresholds)
│   └── agents.yaml                    # Agent config (models, params, prompt paths)
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                          # Shared types & interfaces
│   │   ├── CLAUDE.md
│   │   ├── types.py                   # MarketSnapshot, TradingSignal, PortfolioState, etc.
│   │   ├── interfaces.py             # ABC: BaseLLMAdapter, BaseMarketDataAdapter, etc.
│   │   ├── exceptions.py             # Custom exception classes
│   │   └── constants.py              # Market codes, currencies, magic number removal
│   │
│   ├── data/                          # Market Intelligence Stream (MIS)
│   │   ├── CLAUDE.md
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── krx_adapter.py        # pykrx + KIS API
│   │   │   ├── us_adapter.py         # EODHD + yfinance fallback
│   │   │   ├── crypto_adapter.py     # Binance WebSocket + CCXT
│   │   │   ├── macro_kr_adapter.py   # Bank of Korea ECOS + OpenDART
│   │   │   ├── macro_us_adapter.py   # FRED API
│   │   │   └── news_adapter.py       # Google News RSS + Finnhub + CryptoPanic
│   │   ├── normalizer.py             # Raw -> MarketSnapshot conversion
│   │   ├── cache.py                  # Redis cache layer
│   │   ├── scheduler.py              # APScheduler + EventTrigger
│   │   └── db.py                     # PostgreSQL/TimescaleDB connection & schema
│   │
│   ├── llm/                           # LLM Provider Adapters
│   │   ├── CLAUDE.md
│   │   ├── base.py                   # BaseLLMAdapter implementation helpers
│   │   ├── deepseek_adapter.py       # DeepSeek R1 / V3
│   │   ├── gemini_adapter.py         # Gemini 2.5 Pro
│   │   ├── claude_adapter.py         # Claude Sonnet 4.5
│   │   ├── gpt_adapter.py            # GPT-4o-mini (reference baseline)
│   │   ├── prompt_templates/
│   │   │   ├── single_agent.py       # Single agent prompt
│   │   │   ├── analyst.py            # Multi: fundamental/technical/sentiment/news analysts
│   │   │   ├── researcher.py         # Multi: Bull/Bear researchers
│   │   │   ├── trader.py             # Multi: Trader
│   │   │   ├── risk_manager.py       # Multi: Risk Manager
│   │   │   └── fund_manager.py       # Multi: Fund Manager
│   │   ├── response_parser.py        # LLM response -> TradingSignal parser
│   │   ├── cost_tracker.py           # Per-model token usage & cost tracking
│   │   └── call_logger.py            # Full prompt/response recording
│   │
│   ├── agents/                        # Agent Execution Protocol (AEP)
│   │   ├── CLAUDE.md
│   │   ├── single_agent.py           # Single agent (1 LLM call -> 1 signal)
│   │   ├── multi_agent/
│   │   │   ├── __init__.py
│   │   │   ├── graph.py              # LangGraph workflow definition
│   │   │   ├── analysts.py           # 4 analyst nodes (parallel)
│   │   │   ├── researchers.py        # Bull/Bear debate nodes
│   │   │   ├── trader_node.py        # Trading decision node
│   │   │   ├── risk_node.py          # Risk assessment node
│   │   │   └── fund_manager_node.py  # Final approval node
│   │   └── orchestrator.py           # Parallel execution of all combinations
│   │
│   ├── simulator/                     # Virtual Trading Simulator
│   │   ├── CLAUDE.md
│   │   ├── portfolio.py              # Portfolio state management
│   │   ├── order_engine.py           # Order execution simulation (slippage, fees)
│   │   ├── pnl_calculator.py         # Realized/unrealized P&L, FX conversion
│   │   └── position_tracker.py       # Position history & snapshots
│   │
│   ├── analytics/                     # Performance Analytics Interface (PAI)
│   │   ├── CLAUDE.md
│   │   ├── metrics.py                # Sharpe, Sortino, MDD, Calmar, Win Rate, etc.
│   │   ├── comparator.py             # Statistical comparison between models
│   │   ├── behavioral_profiler.py    # Model personality/behavior analysis
│   │   ├── audit_trail.py            # Decision rationale logging & querying
│   │   └── exporter.py              # CSV/JSON/PDF report generation
│   │
│   └── dashboard/                     # Streamlit Dashboard
│       ├── CLAUDE.md
│       ├── app.py                    # Main Streamlit app
│       ├── pages/
│       │   ├── overview.py           # Overall performance summary
│       │   ├── system_status.py      # Benchmark progress & agent health
│       │   ├── model_comparison.py   # 4-model real-time comparison
│       │   ├── market_view.py        # Per-market (KRX/US/Crypto) detail
│       │   ├── agent_detail.py       # Individual agent decision tracking
│       │   └── cost_monitor.py       # API cost monitoring
│       └── components/
│           ├── equity_curve.py       # Equity curve chart
│           ├── trade_table.py        # Trade history table
│           └── metrics_card.py       # Metrics card widget
│
├── tests/
│   ├── unit/
│   │   ├── test_types.py
│   │   ├── test_normalizer.py
│   │   ├── test_order_engine.py
│   │   ├── test_metrics.py
│   │   └── test_response_parser.py
│   ├── integration/
│   │   ├── test_data_adapters.py     # Live API call tests
│   │   ├── test_llm_adapters.py      # Live LLM call tests
│   │   └── test_full_cycle.py        # 1-cycle E2E
│   └── fixtures/
│       ├── sample_snapshot.json
│       └── sample_signals.json
│
├── scripts/
│   ├── run_benchmark.py              # Benchmark entry point
│   ├── setup_db.py                   # DB schema initialization
│   └── seed_data.py                  # Test data seeding
│
└── docs/
    ├── architecture.md               # This document
    ├── roadmap.md                    # Development roadmap & milestones
    ├── review.md                     # Project review & refinements
    ├── claude-code-prompts.md        # Claude Code prompt guide
    └── benchmarking_protocol.md      # Benchmark execution protocol
```

---

## Module Specifications

### 1. `core/types.py` — System-Wide Shared Types

Full dataclass definitions as shown in the Core Interfaces section above, plus:

- `__post_init__` validation on all dataclasses
- `PortfolioState.total_value` property (cash + sum of positions)
- `MarketSnapshot.to_prompt_summary()` method for token-efficient LLM summaries

### 2. `data/` — Market Intelligence Stream (MIS)

Each adapter implements `BaseMarketDataAdapter`. `normalizer.py` converts market-specific raw data into `MarketSnapshot`.

#### `data/adapters/krx_adapter.py`
```python
class KRXAdapter(BaseMarketDataAdapter):
    """
    Data sources:
    - pykrx: OHLCV, market cap, PER/PBR, investor trading volumes
    - KIS API: Real-time quotes (WebSocket), mock trading support
    - OpenDART: Financial statements, disclosures (dart-fss library)

    Target symbols: Managed in config/markets.yaml
    Default: KOSPI200 representative stocks

    Rate Limits:
    - pykrx: 5 calls/sec (self-throttled with asyncio.Semaphore)
    - KIS: 20 calls/sec (auto token refresh)
    - OpenDART: 10,000 calls/day

    Note: pykrx is synchronous — all calls wrapped with asyncio.to_thread()
    """
```

#### `data/adapters/crypto_adapter.py`
```python
class CryptoAdapter(BaseMarketDataAdapter):
    """
    Data sources:
    - Binance WebSocket: Real-time kline, ticker, depth
    - CCXT: Multi-exchange fallback

    Target symbols: BTC/USDT, ETH/USDT, SOL/USDT + config additions

    Two modes:
    1. stream_mode: WebSocket real-time -> Redis latest price storage
    2. fetch_mode: REST API for recent N klines (fallback)

    Notes:
    - 24/7 operation -> persistent WebSocket + 5-min health check
    - Auto-reconnect on disconnect (max 5 retries, exponential backoff)
    - Exchange down -> automatic CCXT fallback
    """
```

#### `data/adapters/news_adapter.py`
```python
class NewsAdapter:
    """
    Three sources (free/low-cost priority):

    Korean:
    - Google News RSS: https://news.google.com/rss/search?q={symbol}&hl=ko&gl=KR
    - feedparser library, filter to last 1 hour only
    - OpenDART disclosure alerts (already have API)

    US:
    - Finnhub News API: /api/v1/news, /api/v1/company-news
    - Free tier: 60 calls/min

    Crypto:
    - CryptoPanic API: https://cryptopanic.com/api/free/v1/posts/
    - Free tier with filtering limitations

    Processing:
    - Convert each source -> NewsItem
    - relevance_score: symbol name matching (0 or 1, simple)
    - sentiment: 0.0 at this stage (LLM analysts will evaluate)
    - On error: return empty list (news absence must not halt cycles)
    """
```

#### `data/normalizer.py`
```python
class DataNormalizer:
    """
    Responsibilities:
    1. Timestamp -> UTC conversion
    2. Market-specific raw schema -> SymbolData unification
    3. FX conversion (KRW <-> USD <-> USDT)
    4. News/sentiment merge
    5. MarketSnapshot creation & snapshot_id issuance (uuid7)

    Rules:
    - Missing data: forward-fill up to 3 hours
    - Beyond 3 hours: exclude symbol from snapshot + warning log
    """
```

#### `data/scheduler.py`
```python
class MarketScheduler:
    """
    APScheduler AsyncIOScheduler-based market-specific scheduling:
    - KRX during trading hours (09:00~15:30 KST): every 30 minutes
    - US during trading hours (23:30~06:00 KST): every 30 minutes
    - Crypto: every 15 minutes (24/7)

    Each cycle: create_snapshot -> Redis cache + PostgreSQL storage
    """

class EventTrigger:
    """
    Urgent cycle triggers (supplement to regular fixed cycles):

    Conditions:
    1. Price spike: >=3% change from last snapshot close
    2. VIX surge: >=20% increase from previous day
    3. FX spike: USD/KRW >=1% change
    4. Crypto Fear&Greed: <20 or >80

    On trigger: immediately call orchestrator.run_cycle()
    Cooldown: 15 minutes per market after trigger
    Log tag: "EVENT_TRIGGERED: {condition}, {market}, {magnitude}"
    """
```

### 3. `llm/` — LLM Provider Adapters

All 3+1 providers implement `BaseLLMAdapter`. Prompts are split into:
- **System prompt** (static, cacheable)
- **User prompt** (dynamic, changes every cycle)

#### `llm/deepseek_adapter.py`
```python
class DeepSeekAdapter(BaseLLMAdapter):
    """
    Model: deepseek-reasoner (R1)
    SDK: openai.AsyncOpenAI(base_url="https://api.deepseek.com")

    Features:
    - reasoning_content field for CoT extraction -> audit trail
    - Automatic caching ($0.07/M hit vs $0.27/M miss)
    - JSON mode: response_format={"type": "json_object"}

    Error handling:
    - 503 -> 30s wait + retry (max 3)
    - Parse failure -> HOLD signal + error logging
    """
```

#### `llm/gemini_adapter.py`
```python
class GeminiAdapter(BaseLLMAdapter):
    """
    Model: gemini-2.5-pro-preview-06-05
    SDK: google.genai.Client()

    Features:
    - 1M token context -> bulk news/disclosure processing
    - JSON mode: response_mime_type="application/json" + response_schema
    - Grounding: DISABLED for fairness
    """
```

#### `llm/claude_adapter.py`
```python
class ClaudeAdapter(BaseLLMAdapter):
    """
    Model: claude-sonnet-4-5-20250514
    SDK: anthropic.AsyncAnthropic()

    Features:
    - Extended thinking mode -> complex multi-agent reasoning
    - Prompt caching: cache_control={"type": "ephemeral"} on system prompt (90% cost reduction)
    - XML tag-based structured output: <action>, <weight>, <confidence>, <reasoning>
    """
```

#### `llm/gpt_adapter.py`
```python
class GPTAdapter(BaseLLMAdapter):
    """
    Model: gpt-4o-mini (reference baseline)
    SDK: openai.AsyncOpenAI() — no base_url change

    Purpose: Comparison reference for existing research
    Cost: input $0.15/M, output $0.60/M (negligible)
    JSON mode: response_format={"type": "json_object"}
    """
```

#### `llm/response_parser.py`
```python
class ResponseParser:
    """
    LLM response -> TradingSignal conversion

    Parsing strategy (per-model optimization):
    - DeepSeek: JSON mode output, regex fallback
    - Gemini: JSON mode (response_mime_type guarantees)
    - Claude: XML tag-based (<action>, <weight>, <reasoning>)
    - GPT: JSON mode output

    Failure handling:
    - 3 parse failures -> HOLD signal + full raw response logging
    - Abnormal values (weight > 1, confidence < 0) -> clamp + warning
    """
```

#### `llm/cost_tracker.py`
```python
class CostTracker:
    """
    Real-time tracking of all LLM call token usage/cost.

    Recorded fields:
    - model, timestamp, input_tokens, output_tokens
    - cached_tokens (cache hit ratio tracking)
    - cost_usd (per-model pricing applied)
    - latency_ms

    Pricing table:
    - deepseek-reasoner: input $0.55/M, output $2.19/M, cache hit $0.07/M
    - gemini-2.5-pro: input $1.25/M, output $10/M
    - claude-sonnet-4-5: input $3/M, output $15/M, cache hit $0.30/M
    - gpt-4o-mini: input $0.15/M, output $0.60/M

    Aggregations:
    - Per-model daily/weekly/monthly cost
    - Per-architecture cost (single vs multi)
    - Per-market cost (KRX vs US vs Crypto)
    """
```

#### `llm/call_logger.py`
```python
class LLMCallLogger:
    """
    Full prompt/response recording for reproducibility.

    Stores in llm_call_logs table:
    - call_id, signal_id, timestamp, model
    - role: 'single', 'analyst_fundamental', 'researcher_bull', etc.
    - prompt_text: exact text sent to LLM
    - raw_response: full LLM response
    - parsed_success: boolean
    - latency_ms, input_tokens, output_tokens
    """
```

### 4. `agents/` — Agent Execution Protocol (AEP)

#### `agents/single_agent.py`
```python
class SingleAgent:
    """
    Simplest form: 1 LLM call -> 1 TradingSignal

    Input: MarketSnapshot + PortfolioState
    Prompt: Full market data + portfolio status + trading rules
    Output: TradingSignal

    Purpose: Baseline against multi-agent
    """
```

#### `agents/multi_agent/graph.py` — LangGraph Workflow
```
LangGraph StateGraph structure:

    START
      |
    Parallel Analyst Node
      [Fundamental] [Technical] [Sentiment] [News]  (fan-out -> fan-in)
      |
    Bull/Bear Debate Node
      (configurable rounds: 2~3)
      |
    Trader Decision Node
      |
    Risk Manager Node
      (VETO possible -> return to Trader, max 2 retries)
      |
    Fund Manager Node
      (final APPROVE / REJECT)
      |
    END -> TradingSignal

State Schema (TypedDict, NOT dataclass — LangGraph compatibility):
{
    "snapshot": MarketSnapshot,
    "portfolio": PortfolioState,
    "llm_adapter": BaseLLMAdapter,
    "analyst_reports": list[dict],
    "bull_case": str,
    "bear_case": str,
    "debate_log": list[str],
    "trade_proposal": dict,
    "risk_assessment": dict,
    "risk_veto_count": int,
    "final_signal": TradingSignal | None
}

LLM calls per cycle: analysts 4 + researchers 2*N(rounds) + trader 1 + risk 1 + fund_manager 1
= minimum 10 calls/cycle (2 rounds)
```

#### `agents/orchestrator.py`
```python
class BenchmarkOrchestrator:
    """
    Parallel execution of all agent combinations.

    Execution matrix: 4 models x 2 architectures + Buy&Hold = 9 tasks

    Rules:
    - All tasks via asyncio.gather() simultaneously
    - Each task has independent PortfolioState
    - Timeout: single 30s, multi 120s
    - Timeout exceeded -> HOLD + logging

    Error isolation:
    - 1 agent failure -> only that agent gets HOLD, rest continue normally
    - DB save failure -> local file fallback + warning
    - Full cycle failure -> 60s wait then retry
    """
```

### 5. `simulator/` — Virtual Trading Simulator

#### `simulator/order_engine.py`
```python
class OrderEngine:
    """
    Simulation realism factors:

    1. Slippage: Order-volume-based linear slippage (default 0.1%)
    2. Commissions:
       - KRX: buy 0.015%, sell 0.015% + tax 0.18%
       - US: $0 (zero-commission broker assumed)
       - Crypto: 0.1% maker/taker (Binance default)
    3. Fill price: close price at signal time + slippage
    4. Partial fills: not supported (full fill assumed)
    5. Short selling: not supported (long only)
    6. Max position weight: 30% of portfolio
    7. Min cash: 20% reserve

    Initial capital:
    - KRX: 100,000,000 KRW
    - US: $100,000
    - Crypto: $100,000 (USDT)
    """
```

### 6. `analytics/` — Performance Analytics Interface

#### `analytics/metrics.py`
```python
class MetricsEngine:
    """
    Calculated metrics:

    Return metrics:
    - Cumulative Return (CR)
    - Annualized Return (AR)
    - Daily/Weekly Return Distribution

    Risk-adjusted metrics:
    - Sharpe Ratio (rf = per-market risk-free rate)
    - Sortino Ratio (downside volatility only)
    - Calmar Ratio (AR / MDD)
    - Information Ratio (vs buy-and-hold)

    Risk metrics:
    - Maximum Drawdown (MDD)
    - Value at Risk (VaR 95%)
    - Volatility (annualized)

    Trading metrics:
    - Win Rate
    - Profit Factor (gross profit / gross loss)
    - Average Holding Period
    - Trade Frequency

    AI-specific metrics:
    - Signal Accuracy (directional correctness)
    - Confidence Calibration (confidence vs actual performance correlation)
    - Latency (model response time)
    - Cost per Signal (API cost per signal)
    - Cost-Adjusted Alpha (CAA):
      = (agent return - B&H return) / total API cost
      CAA > 0: positive excess return per $1 API cost
      CAA < 0: paying to underperform B&H
    """
```

#### `analytics/comparator.py`
```python
class ModelComparator:
    """
    Statistical comparison:
    - Welch's t-test: pairwise return difference significance
    - Kruskal-Wallis: 3+ model simultaneous comparison (non-parametric)
    - Bootstrap Confidence Interval: 1,000 resamples for CI estimation
    - Rolling window analysis: 3-day windows for stability assessment

    Regime analysis:
    - Auto-classify market regimes: up (>+0.5%), down (<-0.5%), sideways
    - Per-regime model performance decomposition

    Output:
    - Model x Architecture x Market 3D comparison table
    - Win/loss matrix (A beats B count/ratio)
    """
```

#### `analytics/behavioral_profiler.py`
```python
class BehavioralProfiler:
    """
    Model personality/behavior analysis:

    1. Action Distribution: BUY/SELL/HOLD ratios
    2. Conviction Spread: confidence score distribution
    3. Contrarian Score: ratio of signals opposing majority
    4. Regime Sensitivity: behavior change magnitude across regimes
    5. Reasoning Keywords: TF-IDF top-20 keywords per model
    """
```

---

## Database Schema

```sql
-- Core tables
CREATE TABLE market_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    market TEXT NOT NULL,
    data JSONB NOT NULL
);
-- TimescaleDB hypertable on timestamp

CREATE TABLE trading_signals (
    signal_id TEXT PRIMARY KEY,
    snapshot_id TEXT REFERENCES market_snapshots(snapshot_id),
    timestamp TIMESTAMPTZ NOT NULL,
    model TEXT NOT NULL,
    architecture TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    weight FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    reasoning TEXT NOT NULL,
    latency_ms FLOAT NOT NULL,
    token_usage JSONB
);

CREATE TABLE portfolio_states (
    portfolio_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    model TEXT NOT NULL,
    architecture TEXT NOT NULL,
    market TEXT NOT NULL,
    cash FLOAT NOT NULL,
    positions JSONB NOT NULL,
    total_value FLOAT NOT NULL,
    PRIMARY KEY (portfolio_id, timestamp)
);
-- TimescaleDB hypertable on timestamp

CREATE TABLE trades (
    trade_id TEXT PRIMARY KEY,
    signal_id TEXT REFERENCES trading_signals(signal_id),
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity FLOAT NOT NULL,
    price FLOAT NOT NULL,
    commission FLOAT NOT NULL,
    slippage FLOAT NOT NULL,
    realized_pnl FLOAT
);

CREATE TABLE cost_logs (
    log_id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cached_tokens INTEGER DEFAULT 0,
    cost_usd FLOAT NOT NULL,
    latency_ms FLOAT NOT NULL
);

CREATE TABLE llm_call_logs (
    call_id TEXT PRIMARY KEY,
    signal_id TEXT REFERENCES trading_signals(signal_id),
    timestamp TIMESTAMPTZ NOT NULL,
    model TEXT NOT NULL,
    role TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    parsed_success BOOLEAN NOT NULL,
    latency_ms FLOAT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER
);
```

---

## Key Dependencies

```toml
[project]
name = "atlas"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Core
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",

    # Async
    "aiohttp>=3.9",
    "httpx>=0.27",

    # LLM Providers
    "openai>=1.40",          # DeepSeek + GPT compatible (base_url change)
    "anthropic>=0.40",       # Claude
    "google-genai>=1.0",     # Gemini

    # Agent Framework
    "langgraph>=0.2",
    "langchain-core>=0.3",

    # Market Data
    "pykrx>=1.0",            # KRX
    "python-kis>=1.0",       # Korea Investment & Securities
    "fredapi>=0.5",          # FRED
    "ccxt>=4.0",             # Crypto (multi-exchange)
    "python-binance>=1.0",   # Binance WebSocket
    "dart-fss>=0.4",         # OpenDART
    "yfinance>=0.2",         # US stocks fallback
    "feedparser>=6.0",       # News RSS parsing

    # Database
    "asyncpg>=0.29",         # PostgreSQL async
    "redis>=5.0",            # Redis cache
    "sqlalchemy>=2.0",

    # Scheduling
    "apscheduler>=3.10",

    # Analytics & Dashboard
    "pandas>=2.0",
    "numpy>=1.26",
    "scipy>=1.12",
    "scikit-learn>=1.4",     # TF-IDF for behavioral profiling
    "streamlit>=1.30",
    "plotly>=5.18",

    # Utilities
    "python-dotenv>=1.0",
    "structlog>=24.0",       # Structured logging
    "uuid7>=0.1",            # Time-ordered UUID
]
```
