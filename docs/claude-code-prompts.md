# ATLAS - Claude Code Prompt Guide

Copy-paste these prompts into Claude Code sessions, one at a time.
Complete each prompt, verify the checkpoint, then move to the next.
If context exceeds ~50%, use `/clear` and start a new session with the next prompt.

---

## Session Management Rules

- **One module per session.** Don't work on `data/` and `llm/` simultaneously.
- **Session start:** Always read the relevant module's `CLAUDE.md` + `core/types.py` + `core/interfaces.py`.
- **Progress recording:** Update `SCRATCHPAD.md` at project root between sessions.
- **Verification:** Run the checkpoint commands after each prompt before moving on.

---

## Week 1: Foundation

### Prompt 1 — Project Scaffolding

```
ATLAS 프로젝트를 초기화해줘. 다음 파일들을 생성:

1. pyproject.toml — 의존성은 docs/architecture.md 참고
2. .env.example — 필요한 API 키 목록 (값은 비워둬)
3. Makefile — install, db-up, db-init, test, test-unit, run, dashboard 타겟
4. docker-compose.yml — PostgreSQL 16 + TimescaleDB + Redis 7
5. .gitignore — Python + .env + __pycache__ + .venv

디렉토리 구조는 docs/architecture.md의 트리를 그대로 따라.
빈 __init__.py 파일도 모두 생성해줘.
CLAUDE.md 파일들은 이미 있으니 건드리지 마.
```

### Prompt 2 — Core Types

```
src/core/types.py를 구현해줘.

docs/architecture.md의 "Core Interfaces" 섹션에
정의된 dataclass들을 그대로 구현해. 추가로:

- 모든 dataclass에 __post_init__으로 기본 validation 추가
- PortfolioState에 total_value 프로퍼티 추가 (cash + 포지션 합계)
- MarketSnapshot에 to_prompt_summary() 메서드 추가
  → LLM에 보낼 때 토큰 절약용 요약 문자열 반환

완성 후 tests/unit/test_types.py도 작성해줘.
```

### Prompt 3 — Interfaces & Exceptions

```
두 파일을 구현해줘:

1. src/core/interfaces.py
- BaseLLMAdapter (ABC): generate_signal(snapshot, portfolio) -> TradingSignal
- BaseMarketDataAdapter (ABC): fetch_latest() -> dict[str, SymbolData]
- BaseMetricsCalculator (ABC): calculate(portfolio_history) -> dict
- 모든 메서드는 async

2. src/core/exceptions.py
- ATLASBaseError (베이스)
- DataFetchError (API 호출 실패)
- LLMResponseError (LLM 응답 파싱 실패)
- InsufficientFundsError (현금 부족)
- SnapshotStaleError (스냅샷 3시간 초과 오래됨)
- RateLimitError (API rate limit 초과)

각 예외에 context dict 필드 포함해서 디버깅 정보 담을 수 있게.
```

### Prompt 4 — DB Schema

```
src/data/db.py를 구현해줘.

SQLAlchemy 2.0 + asyncpg 기반으로:

테이블:
- market_snapshots: snapshot_id(PK), timestamp, market, data(JSONB)
  → TimescaleDB 하이퍼테이블 (timestamp 기준)
- trading_signals: signal_id(PK), snapshot_id(FK), timestamp, model,
  architecture, action, weight, confidence, reasoning, latency_ms, token_usage(JSONB)
- portfolio_states: portfolio_id, timestamp, model, architecture, market,
  cash, positions(JSONB), total_value
  → TimescaleDB 하이퍼테이블
- trades: trade_id, signal_id(FK), timestamp, symbol, action, quantity,
  price, commission, slippage, realized_pnl
- cost_logs: log_id, timestamp, model, input_tokens, output_tokens,
  cached_tokens, cost_usd, latency_ms
- llm_call_logs: call_id, signal_id(FK), timestamp, model, role,
  prompt_text, raw_response, parsed_success, latency_ms, input_tokens, output_tokens

scripts/setup_db.py도 만들어줘 — 테이블 생성 + 하이퍼테이블 변환.
```

### Prompt 5 — Configuration Files

```
config/ 하위 파일들을 구현해줘.

1. config/settings.py — Pydantic Settings
- .env에서 로드: 모든 API 키, DB URL, Redis URL
- 환경별 분기: dev/prod (기본 dev)
- 각 키에 SecretStr 타입 사용

2. config/markets.yaml
- KRX:
  - 종목: KOSPI200 대표 30종목 (삼성전자, SK하이닉스 등)
  - 수수료: 매수 0.015%, 매도 0.015% + 세금 0.18%
  - 초기자본: 100000000 (1억원)
  - cycle_interval: 30분
- US:
  - 종목: S&P500 상위 20종목 (AAPL, MSFT, NVDA 등)
  - 수수료: 0%
  - 초기자본: 100000 (USD)
  - cycle_interval: 30분
- CRYPTO:
  - 종목: BTC/USDT, ETH/USDT, SOL/USDT
  - 수수료: maker 0.1%, taker 0.1%
  - 초기자본: 100000 (USDT)
  - cycle_interval: 15분

3. config/agents.yaml
- models:
  - deepseek: model_name, base_url, timeout, temperature(0.0)
  - gemini: model_name, timeout, temperature(0.0)
  - claude: model_name, timeout, temperature(0.0), extended_thinking(false)
  - gpt: model_name, timeout, temperature(0.0)
- architectures: [single, multi]
- multi_agent:
  - debate_rounds: 2
  - analyst_timeout: 60s
  - risk_veto_max_retries: 2
```

---

## Week 2: Data Pipeline

### Prompt 6 — KRX Data Adapter

```
src/data/adapters/krx_adapter.py를 구현해줘.

pykrx 라이브러리 사용. pykrx는 동기 라이브러리니까
모든 호출을 asyncio.to_thread()로 감싸줘.

구현할 메서드:
- fetch_latest(): KOSPI200 대표 종목의 당일 OHLCV + 시가총액 + PER/PBR
- fetch_investor_trading(): 투자자별(외국인/기관/개인) 순매수
- fetch_historical(days): 과거 N일 OHLCV

대상 종목은 config/markets.yaml에서 로드.
Rate limit: 초당 5회 제한 (asyncio.Semaphore 사용).
에러 시 3회 재시도 + exponential backoff.
로깅은 structlog 사용.

tests/unit/test_krx_adapter.py도 작성 — mock으로 pykrx 응답 대체.
```

### Prompt 7 — Korean Macro Data

```
src/data/adapters/macro_kr_adapter.py를 구현해줘.

두 가지 소스:
1. 한국은행 ECOS API (키: 환경변수 BOK_API_KEY)
   - 기준금리, 소비자물가지수(CPI), 원/달러 환율
   - 엔드포인트: https://ecos.bok.or.kr/api/

2. OpenDART API (키: 환경변수 OPENDART_API_KEY)
   - dart-fss 라이브러리 사용
   - config/markets.yaml의 KRX 종목들의 최신 재무제표

MacroData dataclass의 kr_base_rate, kr_cpi_yoy, usdkrw 필드를 채워줘.
재무제표 데이터는 SymbolData.extra dict에 넣어.
```

### Prompt 8 — US Market + FRED

```
두 어댑터를 구현해줘:

1. src/data/adapters/us_adapter.py
- EODHD API (키: EODHD_API_KEY) 사용
- S&P500 상위 20종목 EOD 데이터
- fallback: yfinance (EODHD 실패 시)

2. src/data/adapters/macro_us_adapter.py
- fredapi 라이브러리 (키: FRED_API_KEY)
- 시리즈: FEDFUNDS(금리), CPIAUCSL(CPI), VIXCLS(VIX),
  DGS10(10년 국채), DGS2(2년 국채)
- MacroData의 us_fed_rate, us_cpi_yoy, vix 필드 채우기
```

### Prompt 9 — Crypto Adapter

```
src/data/adapters/crypto_adapter.py를 구현해줘.

Binance WebSocket 사용 (python-binance 라이브러리):
- BTC/USDT, ETH/USDT, SOL/USDT
- 실시간 kline(1분봉) 스트림 구독
- 연결 끊김 시 자동 재연결 (최대 5회, exponential backoff)
- 5분마다 health check ping

두 가지 모드:
1. stream_mode: WebSocket으로 실시간 수신 → Redis에 최신 가격 저장
2. fetch_mode: REST API로 최근 N개 kline 조회 (fallback용)

CCXT도 fallback으로 준비:
- Binance 다운 시 CCXT unified API로 전환
- exchange.set_sandbox_mode(True) 옵션 준비

Fear & Greed Index도 수집해서 MacroData.fear_greed_index에 넣어줘.
alternative.me API 사용.
```

### Prompt 10 — Normalizer + Cache + Scheduler + News

```
네 파일을 구현해줘:

1. src/data/normalizer.py
- 각 어댑터의 원본 데이터를 MarketSnapshot으로 변환
- 타임스탬프: 모두 UTC로 변환
- 환율: KRW/USD 교차 변환 (ECOS에서 가져온 환율 사용)
- 누락 데이터: forward-fill 최대 3시간, 초과 시 해당 종목 제외 + 경고 로그
- snapshot_id: uuid7 사용 (시간순 정렬 가능)

2. src/data/cache.py
- Redis async 클라이언트
- set_snapshot(market, snapshot): TTL 자동 적용
  - KRX: 60초, US: 60초, CRYPTO: 10초
- get_latest_snapshot(market) -> MarketSnapshot | None
- set_price(symbol, price): 개별 종목 최신가 (WebSocket용)

3. src/data/scheduler.py
- APScheduler AsyncIOScheduler 사용
- KRX 장중(09:00~15:30 KST): 30분마다 create_snapshot("KRX")
- US 장중(23:30~06:00 KST): 30분마다 create_snapshot("US")
- Crypto: 15분마다 create_snapshot("CRYPTO")
- 스냅샷 생성 → Redis 캐시 + PostgreSQL 저장
- EventTrigger 클래스: 가격 ±3%, VIX +20%, USD/KRW ±1%, Fear&Greed <20 or >80
  → 즉시 임시 사이클 실행, 쿨다운 15분

4. src/data/adapters/news_adapter.py
- 한국: Google News RSS (feedparser), 최근 1시간 뉴스 필터
- 미국: Finnhub News API (무료 60호출/분)
- 크립토: CryptoPanic API (무료 티어)
- 각 소스 → NewsItem 변환
- relevance_score: 종목명 매칭 기반 (0 or 1)
- sentiment: 0.0 (LLM 분석가가 판단)
- 에러 시 빈 리스트 반환 (뉴스 없어도 사이클 계속)
```

---

## Week 3: LLM Adapters

### Prompt 11 — LLM Common Infrastructure

```
네 파일을 구현해줘:

1. src/llm/cost_tracker.py
- 모든 LLM 호출을 감싸는 async context manager
- 기록: model, timestamp, input_tokens, output_tokens,
  cached_tokens, cost_usd, latency_ms
- DB(cost_logs 테이블)에 즉시 저장
- 모델별 단가 테이블 내장:
  - deepseek-reasoner: input $0.55/M, output $2.19/M, cache hit $0.07/M
  - gemini-2.5-pro: input $1.25/M, output $10/M
  - claude-sonnet-4-5: input $3/M, output $15/M, cache hit $0.30/M
  - gpt-4o-mini: input $0.15/M, output $0.60/M

2. src/llm/response_parser.py
- parse_signal(raw_response, model, snapshot_id) -> TradingSignal
- 모델별 전략:
  - DeepSeek: JSON 파싱 우선, 실패 시 regex
  - Gemini: JSON 파싱 (response_mime_type 보장)
  - Claude: XML 태그 추출 (<action>, <weight>, <confidence>, <reasoning>)
  - GPT: JSON 파싱
- 3회 파싱 실패 → HOLD 시그널 반환 + 원본 로깅
- validation: weight 0~1 clamp, confidence 0~1 clamp

3. src/llm/base.py
- BaseLLMAdapter의 공통 로직 구현
- 재시도: 3회, exponential backoff
- 타임아웃: 설정에서 로드
- cost_tracker 자동 적용
- 에러 분류: RateLimitError, LLMResponseError 등

4. src/llm/call_logger.py
- 모든 LLM 호출의 프롬프트 + 응답 전문을 DB(llm_call_logs)에 기록
- async context manager 형태
- cost_tracker와 함께 사용
```

### Prompt 12 — DeepSeek Adapter

```
src/llm/deepseek_adapter.py를 구현해줘.

OpenAI SDK 호환 방식:
- client = AsyncOpenAI(base_url="https://api.deepseek.com", api_key=...)
- 모델: "deepseek-reasoner" (R1)

특이사항:
- response.choices[0].message.reasoning_content에서 CoT 추출
  → TradingSignal.reasoning에 포함
- JSON mode: response_format={"type": "json_object"}
- 캐시: 자동 적용됨 (별도 설정 불필요)

generate_signal() 구현:
1. MarketSnapshot.to_prompt_summary()로 요약 생성
2. system_prompt (트레이딩 규칙 + JSON 출력 형식) + user_prompt (시장 데이터 + 포트폴리오)
3. API 호출 → response_parser로 파싱 → TradingSignal 반환
```

### Prompt 13 — Gemini Adapter

```
src/llm/gemini_adapter.py를 구현해줘.

google-genai SDK 사용:
- client = genai.Client(api_key=...)
- 모델: "gemini-2.5-pro-preview-06-05"

특이사항:
- JSON mode: generation_config에 response_mime_type="application/json" 설정
- 100만 토큰 컨텍스트 → 멀티에이전트에서 전체 분석가 리포트 한번에 전달 가능
- Grounding: 이 프로젝트에서는 비활성화 (공정성)

generate_signal() 구현:
- DeepSeek과 동일한 흐름이지만 SDK가 다름
- genai.types.GenerateContentConfig 사용
- 응답에서 text 추출 → response_parser로 파싱
```

### Prompt 14 — Claude Adapter

```
src/llm/claude_adapter.py를 구현해줘.

anthropic SDK 사용:
- client = AsyncAnthropic(api_key=...)
- 모델: "claude-sonnet-4-5-20250514"

특이사항:
- Prompt caching 적용:
  system prompt 블록에 cache_control={"type": "ephemeral"} 추가
  → 반복 호출 시 90% 비용 절감
- 출력 형식: XML 태그 기반
  <trading_signal>
    <action>BUY</action>
    <weight>0.15</weight>
    <confidence>0.72</confidence>
    <reasoning>...</reasoning>
  </trading_signal>
- extended_thinking: config에서 on/off 가능하게 (기본 off)

generate_signal() 구현:
- system prompt에 캐시 설정 포함
- user prompt에 MarketSnapshot 요약 + 포트폴리오
- XML 태그 기반 파싱
```

### Prompt 15 — GPT Adapter + Prompt Templates

```
두 가지를 구현해줘:

1. src/llm/gpt_adapter.py
- DeepSeek 어댑터와 거의 동일한 구조 (둘 다 OpenAI SDK)
- client = AsyncOpenAI(api_key=...) — base_url 변경 없음
- 모델: "gpt-4o-mini"
- JSON mode: response_format={"type": "json_object"}
- 역할: 기존 연구와의 비교를 위한 참조 베이스라인
- cost_tracker 단가: input $0.15/M, output $0.60/M

2. src/llm/prompt_templates/single_agent.py
함수 형태로 정의:

def build_system_prompt(market: Market) -> str:
  - 역할: 너는 {market} 시장 전문 트레이딩 에이전트다
  - 규칙: 롱 온리, 최대 포지션 비중 30%, 현금 최소 20% 유지
  - 분석 프레임워크: 펀더멘털 + 테크니컬 + 매크로 + 센티먼트 종합
  - 출력 형식: 모델별 분기 (JSON or XML)
  - few-shot 예시 1개 포함

def build_user_prompt(snapshot: MarketSnapshot, portfolio: PortfolioState) -> str:
  - 현재 시각, 시장 상태
  - snapshot.to_prompt_summary() 결과
  - 현재 포트폴리오: 보유 종목, 수량, 평단가, 수익률
  - 잔여 현금 비율
  - "위 정보를 종합하여 트레이딩 시그널을 생성하라"

시장별 뉘앙스 차이 반영:
- KRX: 외국인/기관 수급, 환율 영향 강조
- US: 섹터 로테이션, Fed 정책 강조
- Crypto: 변동성, Fear&Greed, 온체인 지표 강조
```

---

## Week 4: Multi-Agent + Simulator

### Prompt 16 — LangGraph Workflow

```
src/agents/multi_agent/graph.py를 구현해줘.

LangGraph StateGraph 사용.

State (TypedDict):
- snapshot: MarketSnapshot
- portfolio: PortfolioState
- llm_adapter: BaseLLMAdapter
- analyst_reports: list[dict]
- bull_case: str
- bear_case: str
- debate_log: list[str]
- trade_proposal: dict
- risk_assessment: dict
- risk_veto_count: int
- final_signal: TradingSignal | None

그래프 구조:
START → parallel_analysts → aggregate_reports → bull_bear_debate
→ trader_decision → risk_assessment → (conditional: approved?
→ yes: fund_manager → END, no & veto_count < 2: trader_decision,
no & veto_count >= 2: force_hold → END)

parallel_analysts는 LangGraph의 fan-out/fan-in 패턴 사용.
각 노드는 해당 모듈의 함수를 호출.
```

### Prompt 17 — Analyst Nodes

```
src/agents/multi_agent/analysts.py를 구현해줘.

4개 분석가 함수:

1. fundamental_analyst(state) -> state
   - SymbolData의 PER/PBR/시가총액 + extra(재무제표) 분석
   - 프롬프트: "펀더멘털 관점에서 각 종목의 적정가치를 평가하라"
   - 출력: {종목: 평가등급(overvalued/fair/undervalued), 근거}

2. technical_analyst(state) -> state
   - 가격 데이터로 RSI, MACD, 볼린저밴드, 이동평균 계산
   - 기술 지표 계산은 LLM이 아니라 Python(pandas)으로 직접
   - LLM에게는 계산된 지표를 주고 해석만 시킴
   - 출력: {종목: 신호(bullish/neutral/bearish), 근거}

3. sentiment_analyst(state) -> state
   - 뉴스 센티먼트 + Fear&Greed 분석
   - 출력: {시장전반: 센티먼트 스코어, 근거}

4. news_analyst(state) -> state
   - 매크로 데이터(금리, CPI, 환율) + 주요 뉴스 이벤트 평가
   - 출력: {이벤트 리스크 평가, 시장 영향 방향, 근거}

각 분석가는 state["llm_adapter"]를 사용해서 LLM 호출.
프롬프트는 src/llm/prompt_templates/analyst.py에 별도 정의.
```

### Prompt 18 — Researcher + Trader + Risk + Fund Manager

```
나머지 멀티에이전트 노드들을 구현해줘:

1. src/agents/multi_agent/researchers.py
   - bull_bear_debate(state) -> state
   - 4개 분석가 리포트를 받아서:
     - Bull 리서처: 매수 논거 구축
     - Bear 리서처: 매도/관망 논거 구축
     - config의 debate_rounds(기본 2)만큼 반복
   - debate_log에 각 라운드 기록

2. src/agents/multi_agent/trader_node.py
   - trader_decision(state) -> state
   - 디베이트 결과를 종합하여 구체적 매매 제안
   - trade_proposal: {symbol, action, weight, entry_reason}

3. src/agents/multi_agent/risk_node.py
   - risk_assessment(state) -> state
   - 체크: 최대 포지션 비중 30% 초과?, 현금 20% 미만?,
     MDD 임계값 초과?, 변동성 이상 감지?
   - approved: True/False + 거부 사유
   - VETO 시 risk_veto_count 증가

4. src/agents/multi_agent/fund_manager_node.py
   - fund_manager_approval(state) -> state
   - 전체 파이프라인 출력을 최종 검토
   - TradingSignal 생성하여 state["final_signal"]에 저장
```

### Prompt 19 — Orchestrator

```
src/agents/orchestrator.py를 구현해줘.

BenchmarkOrchestrator 클래스:

async def run_cycle(self, snapshots: dict[Market, MarketSnapshot]):
  """
  1사이클 실행:

  1. 시장별로 활성 상태 확인 (장중인지)
  2. 활성 시장의 스냅샷에 대해 9개 에이전트 병렬 실행:
     - DeepSeek Single, Gemini Single, Claude Single, GPT Single
     - DeepSeek Multi, Gemini Multi, Claude Multi, GPT Multi
     - Buy & Hold (아무것도 안 함, 포트폴리오 값만 갱신)
  3. asyncio.gather(*tasks, return_exceptions=True)
     - 개별 실패는 HOLD 처리 + 에러 로깅
  4. 각 시그널을 simulator에 전달 → 주문 실행
  5. 결과를 DB에 저장
  """

async def run_continuous(self):
  """
  연속 실행 루프:
  - scheduler와 연동하여 사이클 반복
  - graceful shutdown (SIGINT/SIGTERM 처리)
  - 사이클 간 상태 persist (Redis + DB)
  """

에러 격리가 핵심:
- 1개 에이전트 타임아웃/에러 → 해당만 HOLD, 나머지 정상
- DB 저장 실패 → 로컬 파일 fallback + 경고
- 전체 사이클 실패 → 60초 대기 후 재시도
```

### Prompt 20 — Virtual Trading Engine

```
src/simulator/ 전체를 구현해줘.

1. portfolio.py
- PortfolioManager: 9개 독립 포트폴리오(8에이전트 + buy-and-hold) 관리
- 초기화: config/markets.yaml의 초기자본으로 생성
- get_state(model, architecture, market) -> PortfolioState
- 매 사이클 스냅샷을 DB(portfolio_states)에 저장

2. order_engine.py
- execute_signal(signal, portfolio) -> Trade | None
- 체결가: signal 시점 close + 슬리피지(기본 0.1%)
- 수수료: config/markets.yaml에서 시장별 로드
- 현금 부족 → 주문 거부 + 로깅
- 최대 포지션 비중 30% 초과 → weight 조정
- Trade 결과를 DB(trades)에 저장

3. pnl_calculator.py
- 실현 P&L: 매도 시 (매도가 - 평단가) × 수량 - 수수료
- 미실현 P&L: (현재가 - 평단가) × 보유수량
- 환율 변환: KRW 포트폴리오는 KRW 기준, USD/USDT는 USD 기준

4. position_tracker.py
- 포지션 변동 히스토리 관리
- 평균 매입단가 업데이트 (추가 매수 시)
- 부분 매도 지원
```

---

## Week 5: Analytics + Dashboard

### Prompt 21 — Metrics Engine

```
src/analytics/metrics.py를 구현해줘.

MetricsEngine 클래스:
- 입력: portfolio_states 시계열 (DB에서 로드)
- 출력: dict (모든 지표)

구현할 지표:
- cumulative_return: (현재가치 / 초기자본) - 1
- annualized_return: (1 + CR) ^ (365/일수) - 1
- sharpe_ratio: (평균일간수익 - rf) / 일간수익표준편차 × √252
  - rf: KRX→한은금리/252, US→3개월T-bill/252, Crypto→0
- sortino_ratio: 하방편차만 사용
- max_drawdown: 고점 대비 최대 하락
- calmar_ratio: AR / |MDD|
- win_rate: 수익 거래 / 전체 거래
- profit_factor: 총이익 / 총손실 (절대값)
- avg_holding_period: 평균 보유 기간
- trade_count: 총 거래 횟수
- cost_per_signal: 총 API 비용 / 총 시그널 수
- signal_accuracy: 시그널 방향과 실제 가격 변동 일치율
- cost_adjusted_alpha: (에이전트 수익률 - B&H 수익률) / 총 API 비용

모든 계산은 pandas 기반. numpy 연산 최적화.
```

### Prompt 22 — Model Comparison + Behavioral Profiler

```
두 파일을 구현해줘:

1. src/analytics/comparator.py

ModelComparator 클래스:
- pairwise_test(model_a, model_b, metric)
  - Shapiro-Wilk 정규성 검정
  - 정규: Welch's t-test / 비정규: Mann-Whitney U test
  - p-value + effect size(Cohen's d) 반환
- multi_model_test(metric)
  - ANOVA or Kruskal-Wallis + Dunn's post-hoc
- generate_comparison_matrix()
  - 4모델 × 2아키텍처 × 3시장 매트릭스
- regime_analysis()
  - 상승(>+0.5%), 하락(<-0.5%), 횡보 자동 분류
  - 국면별 성과 분해
- rolling_window_analysis(window_days=3)
  - 윈도우별 성과 측정, 윈도우 간 분산으로 안정성 평가
- bootstrap_confidence_interval(n_resamples=1000)
  - 일간 수익률 복원추출, 95% CI 추정

2. src/analytics/behavioral_profiler.py

BehavioralProfiler 클래스:
- action_distribution(model, architecture): BUY/SELL/HOLD 비율
- conviction_analysis(model, architecture): confidence 평균/표준편차/분포
- contrarian_score(model, architecture): 다수결과 반대인 비율
- regime_sensitivity(model, architecture): 상승장/하락장 행동 변화폭
- reasoning_keywords(model, architecture): TF-IDF 상위 20 키워드
```

### Prompt 23 — Streamlit Dashboard

```
src/dashboard/ 전체를 구현해줘.

app.py:
- 사이드바에 6개 페이지 네비게이션
- 자동 새로고침 30초
- DB 연결 + Redis 연결

pages/overview.py:
- 상단: 핵심 지표 카드 9장 (각 에이전트 CR)
- 중단: 9개 자산곡선 Plotly 라인차트 (인터랙티브)
  - 색상: DeepSeek=#4CAF50, Gemini=#4285F4, Claude=#D97706, GPT=#FF6B6B, BuyHold=#9E9E9E
  - Single=실선, Multi=점선
- 하단: 최근 거래 내역 테이블 (최신 20건)

pages/system_status.py:
- 벤치마크 시작 시각, 경과 시간, 총 사이클 수
- 8개 에이전트 상태 표시등 (정상/경고/에러)
- 최근 에러 로그 5건
- 다음 사이클 카운트다운 타이머
- 시장별 활성 상태 (장중/장외)
- API 호출 성공률 차트

pages/model_comparison.py:
- 4×2 히트맵: 행=모델, 열=아키텍처, 값=선택 지표(드롭다운)
- 레이더 차트: 6개 지표 동시 비교
- 시장별 탭 분리

pages/market_view.py:
- KRX/US/Crypto 탭
- 시장별 가격 차트 + 에이전트 매매 시점 마커
- 포지션 현황 파이 차트

pages/agent_detail.py:
- 에이전트 선택 드롭다운
- 의사결정 타임라인 (시간축에 BUY/SELL/HOLD 마커)
- 각 시그널 클릭 → reasoning 전문 표시
- 멀티에이전트: Bull/Bear 디베이트 로그 펼쳐보기
- Agent Personality 섹션: 행동 프로파일링 결과 시각화

pages/cost_monitor.py:
- 모델별 누적 비용 라인차트
- 시그널당 평균 비용 바차트
- 토큰 사용량 breakdown (input/output/cached)
- Cost-Adjusted Alpha 비교 차트
```

### Prompt 24 — Benchmark Run Script

```
scripts/run_benchmark.py를 구현해줘.

CLI 엔트리포인트:
- python scripts/run_benchmark.py --markets KRX,US,CRYPTO --duration 72h
- python scripts/run_benchmark.py --markets CRYPTO --duration 24h --dry-run
- python scripts/run_benchmark.py --phase 1 (Crypto only, 72h)
- python scripts/run_benchmark.py --phase 2 (all markets, 2 weeks)

옵션:
- --markets: 활성화할 시장 (콤마 구분)
- --duration: 벤치마크 기간 (시간 단위)
- --dry-run: 1사이클만 실행 후 종료
- --resume: 중단된 벤치마크 이어서 실행
- --phase: 사전 정의된 실행 프로파일

실행 흐름:
1. 설정 로드 + DB 연결 + Redis 연결
2. 9개 포트폴리오 초기화 (또는 --resume 시 복원)
3. 스케줄러 시작
4. 메인 루프: orchestrator.run_continuous()
5. duration 도달 또는 SIGINT → graceful shutdown
6. 최종 리포트 생성 (analytics/exporter.py)
```

---

## Troubleshooting Prompts

### API Errors

```
{모델명} API에서 계속 429(Rate Limit) 에러가 나와.
해당 어댑터의 rate limiting 로직을 확인하고:
1. 현재 호출 빈도 로그 분석
2. backoff 간격을 늘리거나 호출 빈도를 줄여줘
3. 가능하면 배치 처리로 변경
```

### Data Consistency

```
KRX 데이터가 간헐적으로 누락돼.
data/normalizer.py의 forward-fill 로직이 제대로 작동하는지 확인하고,
누락 빈도를 로깅하는 모니터링 코드를 추가해줘.
현재 누락률이 몇 %인지도 출력해줘.
```

### Performance Optimization

```
1사이클 실행 시간이 너무 길어 (목표: 2분 이내).
병목 지점을 찾아줘:
1. 각 단계별 실행 시간 프로파일링 추가
2. LLM 호출이 순차적으로 되고 있으면 병렬로 변경
3. DB 쿼리가 느리면 인덱스 추가
```

### Multi-Agent Debugging

```
{모델} 멀티에이전트에서 Risk Manager가 매번 VETO를 해.
risk_node.py의 판단 기준을 확인하고:
1. 최근 10사이클의 risk_assessment 로그를 보여줘
2. VETO 사유 분석
3. 임계값이 너무 보수적이면 조정 제안해줘
```

### Dashboard Issues

```
Streamlit 대시보드에서 {페이지}가 로딩이 느려.
@st.cache_data 적용 여부 확인하고,
무거운 쿼리를 최적화해줘.
DB 쿼리에 적절한 시간 범위 필터를 추가하는 것도 검토해줘.
```

### Benchmark Result Report

```
{N}시간 벤치마크가 완료됐어.
analytics/exporter.py를 실행해서 최종 리포트를 생성해줘.

리포트 내용:
1. 요약: 각 에이전트 최종 CR, Sharpe, MDD, CAA
2. 승자: 시장별 + 전체 최고 성과 조합
3. 통계적 유의성: pairwise t-test 결과
4. 비용 효율: 시그널당 비용 대비 성과
5. 행동 프로파일: 모델별 트레이딩 성격 비교
6. 핵심 발견: 싱글 vs 멀티 차이, 모델별 특성
7. 시장 국면별 성과 분해

CSV + JSON + 마크다운 리포트로 출력.
```
