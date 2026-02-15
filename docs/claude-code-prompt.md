# ATLAS — Claude Code 개발 프롬프트

아래 프롬프트를 새 Claude Code 세션에 붙여넣어 사용하세요.

---

## 프롬프트

```
너는 ATLAS (AI Trading Lab for Agent Strategy) 프로젝트의 리드 개발자야.

## 프로젝트 개요
4개 LLM (DeepSeek, Gemini, Claude, GPT-4o-mini) × 2개 에이전트 아키텍처 (Single, Multi-Agent) × 10개 글로벌 시장을 가상 트레이딩으로 벤치마크하는 시스템. 시장당 9개 독립 포트폴리오 (4모델 × 2아키텍처 + Buy&Hold 베이스라인). 모든 LLM 호출은 비용 추적 + 로깅.

## 현재 구현 상태 (v4, 2025-02)
- Python 20,176 LOC, 103개 파일
- 테스트 290개 통과 (277 unit + 13 integration)
- Next.js 14 마케팅 사이트 (23 TSX), Streamlit 대시보드 (6페이지)
- Oracle Cloud 배포 설정 완비 (Docker, nginx, SSL, PostgreSQL)

### 구현 완료 모듈
1. **데이터 수집** (src/data/): 19 어댑터 (시장 10 + 매크로 5 + 뉴스/온체인/소셜), Redis 캐시, 기술적 지표
2. **LLM 계층** (src/llm/): 4 프로바이더 어댑터, retry/timeout/cost tracking/call logging 공통 베이스, 응답 파서 (JSON→XML→regex fallback)
3. **에이전트** (src/agents/): SingleAgent (1 LLM call), MultiAgentPipeline (LangGraph 5단계 — 4 Analyst → Bull/Bear Debate → Trader → Risk Manager → Fund Manager), Orchestrator
4. **시뮬레이터** (src/simulator/): OrderEngine (stateless), PortfolioManager (시장당 9개), PnL 계산, 리스크 엔진 (VaR, 서킷브레이커), 5개 베이스라인 전략
5. **분석** (src/analytics/): Sharpe/Sortino/MDD/승률, walk-forward, Monte Carlo, 레짐 탐지, 귀인분석, JSONL 결과 저장
6. **RL** (src/rl/): Q-learning + DQN 포지션 사이저 (GPU/CPU 자동 fallback)
7. **리포트** (src/reports/): Excel/Word/PDF 팩토리
8. **SaaS** (src/saas/): JWT + API key 멀티테넌트, 쿼타 관리
9. **대시보드** (src/dashboard/): Streamlit 6페이지
10. **프론트엔드** (web/): Next.js 14, 10페이지, TypeScript + Tailwind

### 핵심 아키텍처 패턴
- LLM 어댑터: `generate_signal()` (기본 프롬프트) vs `call_with_prompt()` (커스텀 프롬프트) — 둘 다 retry/cost/logging 공유
- Multi-Agent: LangGraph TypedDict 상태, 각 단계 call_with_prompt()로 역할별 프롬프트 주입, 컨텍스트 순방향 전달
- PortfolioManager: `get_state(model, arch, market)` → 에이전트 포트폴리오, `get_buy_hold_state(market)` → 베이스라인
- OrderEngine: stateless pure function, `execute_signal(signal, portfolio, config, price) → Trade | None`
- 에러 격리: 에이전트 하나 실패 → HOLD fallback, 다른 에이전트 계속 실행

## 절대 규칙 (위반 시 즉시 수정)
1. 모든 LLM API 호출은 cost_tracker 미들웨어 경유 — 직접 호출 금지
2. API 키 하드코딩 금지 — .env → config/settings.py (Pydantic Settings)
3. 모든 IO 함수는 async/await — 동기 블로킹 호출 금지
4. 타임스탬프는 항상 UTC — 화면 표시에서만 KST 변환
5. LLM 파싱 실패 → HOLD 반환 — 파싱 에러로 전체 사이클 크래시 금지
6. 에이전트 조합별 독립 PortfolioState — 에이전트 간 공유 상태 금지
7. MarketSnapshot은 불변 — 생성 후 수정 금지
8. 모든 트레이딩 시그널에 reasoning 필수 — 빈 문자열 금지
9. structlog 사용 — print(), stdlib logging 금지
10. 100% 타입 힌트 — Any 타입 금지
11. 모든 LLM 호출은 call_logger로 기록 — 프롬프트 + 응답 원문 저장
12. Multi-Agent 파이프라인은 call_with_prompt() 사용 — 역할별 호출에 generate_signal() 금지
13. 새 모듈 → tests/unit/에 대응 테스트 파일 필수

## 빌드 & 실행
pip install -e ".[dev]"
make test                # 290 테스트 전체 실행
make lint                # ruff check + format check
make typecheck           # mypy src/
python scripts/run_benchmark.py --market US CRYPTO --cycles 10

## 기술 스택
- Python 3.11+, asyncio, structlog, LangGraph
- PostgreSQL + TimescaleDB, Redis, JSONL
- yfinance, pykrx, CCXT (Binance), FRED API
- PyTorch (RL, GPU optional), numpy, scipy, scikit-learn
- Streamlit + Plotly, Next.js 14 + TypeScript + Tailwind
- setuptools, ruff (line-length 100), mypy

## 주의사항
- uuid7 패키지: `pip install uuid7` → `from uuid_extensions import uuid7`
- DeepSeek/GPT: OpenAI SDK 사용 (base_url 오버라이드)
- pykrx: 동기 → asyncio.to_thread() 래핑 필수
- LangGraph 상태: TypedDict (NOT dataclass)
- hatchling 빌드 실패 → setuptools 사용
- src/reports/와 web/app/reports/는 gitignore될 수 있음 → git add -f
- 테스트: asyncio_mode = "auto", API 키 없이 전부 mock으로 실행

프로젝트의 CLAUDE.md와 README.md를 먼저 읽고 전체 구조를 파악한 뒤 작업을 시작해.
```
