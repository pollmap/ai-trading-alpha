# ATLAS v2 Enhancement Design — 극한 보완 설계서

> 현재 ATLAS v1의 한계를 분석하고, 세계 수준의 AI 트레이딩 벤치마크 시스템으로
> 업그레이드하기 위한 설계. 2024-2026년 최신 논문/프레임워크 기반.

---

## 목차

1. [현재 시스템 진단: 치명적 한계 8가지](#1-현재-시스템-진단)
2. [Tier 1: 핵심 보완 (신뢰성 기반)](#2-tier-1-핵심-보완)
3. [Tier 2: 경쟁력 차별화](#3-tier-2-경쟁력-차별화)
4. [Tier 3: 프로덕션급 고도화](#4-tier-3-프로덕션급-고도화)
5. [Tier 4: 차세대 로드맵](#5-tier-4-차세대-로드맵)
6. [신규 디렉터리 구조](#6-신규-디렉터리-구조)
7. [구현 우선순위 & 예상 일정](#7-구현-우선순위)
8. [참고 논문/프레임워크](#8-참고-문헌)

---

## 1. 현재 시스템 진단

### 1.1 LLM은 Buy&Hold도 못 이긴다

**근거**: StockBench (arXiv:2510.02209, 2025) — GPT-5, Claude 4, Qwen3 등 평가 결과
대부분의 LLM 에이전트가 단순 Buy&Hold 수익률을 넘지 못함.

**ATLAS의 문제**: Buy&Hold 베이스라인 1개만 있음. Equal-Weight Rebalancing,
Momentum, Mean-Reversion 등 다양한 naive baseline 없이는 벤치마크 의미 없음.

### 1.2 하락장 맹점 (Bearish Market Blindness)

LLM은 학습 데이터에 상승 내러티브가 더 많아서 구조적으로 낙관 편향.
ATLAS는 시장 레짐(상승/하락/횡보)별 성과 분리 평가가 없음.

### 1.3 숫자 추론 실패

Agent Trading Arena (arXiv:2502.17967, 2025) — LLM은:
- 절대값에 집착, 변화율 무시
- 최근 데이터에 과도하게 가중
- 긴 시계열 추론 불가

ATLAS의 `to_prompt_summary()`가 변화율을 포함하지만, 이런 실패를 감지/교정하는
메커니즘 없음.

### 1.4 전화기 효과 (Telephone Effect)

TradingAgents (arXiv:2412.20138, 2025) — 멀티 에이전트에서 자연어만으로 소통하면
"전화기 게임"처럼 정보가 왜곡됨. 단계가 깊어질수록 초기 정보 손실.

ATLAS의 5단계 파이프라인이 이 문제에 취약. 구조화된 데이터+자연어 하이브리드
통신이 필요.

### 1.5 환각(Hallucination) 리스크

2025년 기준 최선: Gemini 2.0 Flash 0.7% 환각률. 1000번 호출에 7번 거짓 생성.
금융에서는 존재하지 않는 뉴스/실적/규제를 근거로 매매할 수 있음.

ATLAS는 응답 파싱 실패 시 HOLD 처리하지만, "파싱은 성공했지만 내용이 허위"인
경우를 잡지 못함.

### 1.6 자기 반성(Self-Reflection) 없음

각 사이클이 독립적. 과거 매매의 성공/실패를 학습하지 않음.
TradingGroup (arXiv:2508.17565, 2025)은 자기 반성 메커니즘으로 성과 향상 증명.

### 1.7 리스크 관리가 프롬프트 수준

Risk Manager가 프롬프트로 "5가지 체크"를 하라고 지시하지만, 이는 LLM의
판단에 의존. 실제 VaR 계산, 상관관계 모니터링, 서킷 브레이커 같은
**하드코딩된 리스크 인프라** 없음.

### 1.8 기술적 지표 미제공

프롬프트에서 "테크니컬 분석하라"고 하지만, 실제로 RSI, MACD, 볼린저 밴드 등을
계산해서 제공하지 않음. LLM이 OHLCV 원시 데이터만 보고 "기술적 분석"을
시도하게 됨 → 환각 위험 극대화.

---

## 2. Tier 1: 핵심 보완 (신뢰성 기반)

> 이것들 없으면 벤치마크로서 신뢰성 없음

### 2.1 기술적 지표 사전 계산 엔진

```
신규 모듈: src/data/indicators.py
```

**설계**:
```python
@dataclass
class TechnicalIndicators:
    """사전 계산된 기술적 지표. LLM에게 직접 제공."""
    symbol: str
    # 모멘텀
    rsi_14: float | None          # RSI (14일)
    rsi_7: float | None           # RSI (7일, 단기)
    # 추세
    sma_20: float | None          # 20일 단순이동평균
    sma_50: float | None
    sma_200: float | None
    ema_12: float | None          # 12일 지수이동평균
    ema_26: float | None
    # MACD
    macd_line: float | None       # EMA12 - EMA26
    macd_signal: float | None     # MACD의 9일 EMA
    macd_histogram: float | None  # MACD - Signal
    # 볼린저 밴드
    bb_upper: float | None        # 20일 SMA + 2σ
    bb_middle: float | None       # 20일 SMA
    bb_lower: float | None        # 20일 SMA - 2σ
    bb_width: float | None        # (upper - lower) / middle
    # 거래량
    volume_sma_20: float | None   # 20일 평균 거래량
    volume_ratio: float | None    # 현재 거래량 / 20일 평균
    # 변동성
    atr_14: float | None          # Average True Range (14일)
    historical_volatility_20: float | None  # 20일 역사적 변동성


class IndicatorEngine:
    """OHLCV 시계열로부터 기술적 지표 일괄 계산.

    pandas/numpy 기반. LLM이 직접 계산하는 것을 방지.
    계산된 지표는 MarketSnapshot에 포함되어 프롬프트에 주입.
    """

    def calculate(self, ohlcv_df: pd.DataFrame) -> TechnicalIndicators:
        """최소 200일 OHLCV 필요. 부족 시 None 반환."""
        ...
```

**MarketSnapshot 확장**:
```python
@dataclass
class MarketSnapshot:
    # ... 기존 필드 ...
    indicators: dict[str, TechnicalIndicators] = field(default_factory=dict)

    def to_prompt_summary(self) -> str:
        # indicators 섹션 추가
        # === Technical Indicators ===
        # 005930: RSI(14)=55.2 MACD=+120 BB=상단근접 Vol비율=1.3x
```

### 2.2 다중 베이스라인 시스템

```
신규 모듈: src/simulator/baselines.py
```

**설계**:
```python
class BaselineStrategy(ABC):
    """모든 베이스라인 전략의 인터페이스."""
    @abstractmethod
    def generate_signals(self, snapshot: MarketSnapshot, portfolio: PortfolioState) -> list[TradingSignal]: ...

class BuyAndHoldBaseline(BaselineStrategy):
    """초기 균등 매수 후 보유. 현재 구현 확장."""

class EqualWeightRebalanceBaseline(BaselineStrategy):
    """매 사이클 균등 비중 리밸런싱."""

class MomentumBaseline(BaselineStrategy):
    """최근 N일 수익률 상위 K개 종목 매수. 하위 K개 매도."""

class MeanReversionBaseline(BaselineStrategy):
    """RSI 과매도 매수, 과매수 매도."""

class RandomBaseline(BaselineStrategy):
    """무작위 매수/매도. 통계적 유의성 검정용."""
```

### 2.3 환각 감지 레이어

```
신규 모듈: src/llm/hallucination_detector.py
```

**설계**:
```python
class HallucinationDetector:
    """LLM 응답을 실제 데이터와 교차 검증.

    검증 항목:
    1. 언급된 종목이 실제 snapshot에 존재하는가?
    2. 언급된 가격이 실제 가격과 ±5% 이내인가?
    3. 언급된 뉴스가 실제 news 리스트에 존재하는가?
    4. 수치적 주장이 실제 데이터와 일치하는가?
       (예: "RSI가 30 이하" → 실제 RSI 확인)
    """

    def validate(
        self,
        signal: TradingSignal,
        snapshot: MarketSnapshot,
        raw_response: str,
    ) -> HallucinationReport:
        """Returns report with confidence_penalty and flagged_claims."""
        ...

@dataclass
class HallucinationReport:
    is_clean: bool
    flagged_claims: list[FlaggedClaim]
    confidence_penalty: float  # 0.0 ~ 1.0, 신뢰도에서 차감
    recommendation: str  # "proceed" | "reduce_weight" | "reject_to_hold"

@dataclass
class FlaggedClaim:
    claim_text: str
    claim_type: str  # "price" | "indicator" | "news" | "ticker"
    expected_value: str
    actual_value: str
    severity: str  # "minor" | "major" | "critical"
```

### 2.4 하드코딩 리스크 관리 인프라

```
신규 모듈: src/simulator/risk_engine.py
```

**설계 (프롬프트 아닌 코드로 강제)**:
```python
class RiskEngine:
    """프롬프트 기반 Risk Manager를 보완하는 하드코딩 리스크 엔진.

    LLM이 뭐라고 하든 이 엔진의 판단이 최종.
    """

    def __init__(self, config: RiskConfig): ...

    # ── Pre-Trade Checks (매매 전) ──
    def check_position_limit(self, trade: Trade, portfolio: PortfolioState) -> RiskCheck: ...
    def check_cash_reserve(self, trade: Trade, portfolio: PortfolioState) -> RiskCheck: ...
    def check_daily_loss_limit(self, portfolio: PortfolioState) -> RiskCheck: ...
    def check_correlation_exposure(self, trade: Trade, portfolio: PortfolioState) -> RiskCheck: ...

    # ── Portfolio-Level Monitors (실시간) ──
    def calculate_var(self, portfolio: PortfolioState, returns: list[float], confidence: float = 0.95) -> float: ...
    def check_drawdown_circuit_breaker(self, portfolio: PortfolioState) -> bool: ...
    def check_volatility_regime(self, market_data: MarketSnapshot) -> VolatilityRegime: ...

    # ── 종합 판단 ──
    def evaluate(self, trade: Trade, portfolio: PortfolioState, snapshot: MarketSnapshot) -> RiskDecision: ...

@dataclass
class RiskConfig:
    max_position_weight: float = 0.30
    min_cash_ratio: float = 0.20
    daily_loss_limit_pct: float = 0.05      # 일일 최대 손실 5%
    drawdown_circuit_breaker_pct: float = 0.15  # 15% DD시 전체 매매 중단
    var_confidence: float = 0.95
    max_sector_exposure: float = 0.50       # 섹터 집중도 50%
    max_correlation_overlap: float = 0.70   # 상관계수 0.7 이상 종목 합산 제한

@dataclass
class RiskDecision:
    approved: bool
    checks: list[RiskCheck]
    portfolio_var: float
    drawdown_status: str  # "normal" | "warning" | "circuit_breaker"
    override_action: Action | None  # 강제 HOLD 등
```

### 2.5 레짐 감지 & 레짐별 평가

```
신규 모듈: src/analytics/regime_detector.py
```

**설계**:
```python
class MarketRegime(str, Enum):
    BULL = "bull"           # 상승 추세
    BEAR = "bear"           # 하락 추세
    SIDEWAYS = "sideways"   # 횡보
    HIGH_VOL = "high_vol"   # 고변동성 (방향 무관)
    CRASH = "crash"         # 급락 (VIX 스파이크 + 대폭락)

class RegimeDetector:
    """시장 레짐을 실시간으로 분류.

    방법론:
    1. Hidden Markov Model (HMM) - 3 state (bull/bear/sideways)
    2. VIX 기반 변동성 레짐
    3. 이동평균 크로스오버 (SMA50 vs SMA200)
    4. 드로다운 기반 crash 감지

    에이전트 프롬프트에 현재 레짐 정보 주입.
    분석 시 레짐별 성과 분해.
    """

    def detect(self, prices: pd.Series, vix: float | None = None) -> MarketRegime: ...
    def get_regime_history(self, prices: pd.Series) -> list[tuple[datetime, MarketRegime]]: ...

class RegimeAnalyzer:
    """레짐별 성과 분해 분석.

    출력 예시:
    | Regime   | DeepSeek/Single | Claude/Multi | Buy&Hold |
    |----------|----------------|-------------|----------|
    | Bull     | +15.2%         | +18.7%      | +20.1%   |
    | Bear     | -3.1%          | -1.5%       | -12.4%   |
    | Sideways | +2.1%          | +3.4%       | +0.8%    |

    핵심 인사이트: 하락장에서 LLM이 Buy&Hold 대비 얼마나 방어하는가?
    """

    def analyze_by_regime(
        self, portfolio_values: dict[str, list[float]],
        regime_history: list[tuple[datetime, MarketRegime]],
    ) -> dict[MarketRegime, dict[str, PerformanceMetrics]]: ...
```

---

## 3. Tier 2: 경쟁력 차별화

### 3.1 자기 반성 (Self-Reflection) 메커니즘

```
신규 모듈: src/agents/reflection.py
```

**설계**:
```python
class AgentReflector:
    """과거 N 사이클의 매매 결과를 분석하여 프롬프트에 주입.

    TradingGroup (arXiv:2508.17565) 방법론 채택.

    주입 예시:
    === Recent Performance Review (last 10 cycles) ===
    - Win rate: 40% (4/10)
    - Average loss on losing trades: -3.2%
    - Observation: You have been too aggressive on BUY signals
      in high-volatility periods. 3 of 4 losses were BUY during VIX>25.
    - Recommendation: Increase HOLD tendency when VIX > 25.
    """

    def generate_reflection(
        self,
        recent_signals: list[TradingSignal],
        recent_outcomes: list[TradeOutcome],
        current_regime: MarketRegime,
    ) -> str:
        """과거 N개 매매의 성공/실패를 자연어 요약.
        프롬프트의 시스템 메시지에 추가."""
        ...

@dataclass
class TradeOutcome:
    signal: TradingSignal
    entry_price: float
    exit_price: float | None  # None이면 아직 보유중
    realized_pnl: float
    holding_periods: int  # 사이클 수
    was_correct: bool  # 방향성 적중 여부
```

### 3.2 멀티모델 합의 (Ensemble Consensus)

```
신규 모듈: src/agents/consensus.py
```

**설계**:
```python
class ConsensusEngine:
    """4개 LLM의 신호를 앙상블하여 합의 신호 생성.

    ATLAS의 고유한 강점: 4개 서로 다른 LLM이 동시에 분석.
    이를 활용한 환각 감지 + 앙상블 신호.

    합의 규칙:
    1. 4개 중 3개 이상 같은 방향 → High Confidence 합의 신호
    2. 4개 중 2개 이상 같은 방향 → Medium Confidence
    3. 4개 모두 다른 방향 → HOLD (불일치)

    환각 감지:
    - 3개가 HOLD인데 1개만 강한 BUY → 환각 의심
    - reasoning에서 다른 3개가 언급하지 않는 "사실"을 근거로 → 플래그
    """

    def build_consensus(
        self,
        signals: dict[ModelProvider, TradingSignal],
    ) -> ConsensusSignal: ...

    def detect_outlier(
        self,
        signals: dict[ModelProvider, TradingSignal],
    ) -> list[OutlierFlag]: ...

@dataclass
class ConsensusSignal:
    action: Action
    confidence: float
    agreement_ratio: float  # 0.25 ~ 1.0
    contributing_models: list[ModelProvider]
    dissenting_models: list[ModelProvider]
    reasoning: str  # 합의 근거 요약

# 앙상블 포트폴리오: 9번째가 아닌 10번째 포트폴리오로 추가
# 4 models × 2 archs + Buy&Hold + Ensemble = 10 portfolios per market
```

### 3.3 온체인 분석 (Crypto 전용)

```
신규 모듈: src/data/adapters/onchain_adapter.py
```

**설계**:
```python
class OnChainAdapter(BaseMarketDataAdapter):
    """블록체인 온체인 데이터 수집.

    데이터 소스:
    - Glassnode API (고래 이동, 거래소 유입/유출)
    - Dune Analytics (DEX 거래량, TVL)
    - Whale Alert (대규모 전송 알림)

    핵심 지표:
    """

    async def fetch_latest(self) -> OnChainData: ...

@dataclass
class OnChainData:
    # 거래소 흐름
    exchange_inflow_btc: float      # 거래소 유입량 (매도 압력 신호)
    exchange_outflow_btc: float     # 거래소 유출량 (장기 보유 신호)
    exchange_netflow_btc: float     # 순유입 (양수 = 매도 압력)

    # 고래 활동
    whale_transactions_24h: int     # 24h 대규모 전송 횟수 (>$1M)
    whale_accumulation: bool        # 고래 순매수 여부

    # 네트워크 활동
    active_addresses_24h: int       # 활성 주소 수
    nvt_ratio: float               # Network Value to Transactions ratio

    # DeFi
    total_value_locked: float       # TVL (DeFi 건전성 지표)
    dex_volume_24h: float          # DEX 거래량

    # 펀딩레이트
    funding_rate_btc: float        # 선물 펀딩레이트 (양수=롱 과열)
    funding_rate_eth: float
    open_interest: float           # 미결제약정
```

### 3.4 소셜 센티먼트 파이프라인

```
신규 모듈: src/data/adapters/social_adapter.py
```

**설계**:
```python
class SocialSentimentAdapter:
    """소셜 미디어 센티먼트 수집.

    소스:
    - LunarCrush API (크립토 소셜 지표)
    - Reddit API (r/wallstreetbets, r/cryptocurrency 등)
    - Stocktwits API (미국 주식)

    지표:
    """

    async def fetch_latest(self, market: Market) -> SocialSentiment: ...

@dataclass
class SocialSentiment:
    symbol: str
    social_volume_24h: int          # 소셜 미디어 언급량
    social_volume_change_pct: float # 24h 변화율
    sentiment_score: float          # -1.0 ~ +1.0
    bullish_ratio: float           # 긍정 비율
    trending_rank: int | None      # 트렌딩 순위
    top_keywords: list[str]        # 주요 키워드
    influencer_sentiment: float    # 인플루언서 의견 (가중 점수)
```

### 3.5 Walk-Forward 분석 프레임워크

```
신규 모듈: src/analytics/walk_forward.py
```

**설계**:
```python
class WalkForwardAnalyzer:
    """Walk-Forward Analysis (WFA) — 롤링 윈도우 검증.

    과적합 방지를 위한 gold standard 방법론.
    (Interactive Brokers 2025 가이드라인 준용)

    프로세스:
    1. 데이터를 N개 윈도우로 분할
    2. 각 윈도우: In-Sample(학습) + Out-of-Sample(검증)
    3. IS에서의 성과 vs OOS에서의 성과 비교
    4. OOS 성과만 최종 보고

    핵심 지표:
    - WFA Efficiency = OOS Return / IS Return (0.6-0.75가 정상)
    - 0.6 미만: 과적합 의심
    - 0.75 이상: 견고한 전략
    """

    def run(
        self,
        portfolio_values: list[float],
        window_size: int = 30,
        oos_size: int = 10,
    ) -> WalkForwardResult: ...

class MonteCarloSimulator:
    """몬테카를로 시뮬레이션으로 신뢰구간 추정.

    수익률 시계열을 무작위 재배열하여 N회 시뮬레이션.
    실제 성과가 우연의 범위를 벗어나는지 검증.
    """

    def simulate(
        self,
        returns: list[float],
        n_simulations: int = 10_000,
        confidence_level: float = 0.95,
    ) -> MonteCarloResult: ...

@dataclass
class MonteCarloResult:
    actual_return: float
    simulated_mean: float
    simulated_std: float
    confidence_interval: tuple[float, float]  # (lower, upper)
    percentile_rank: float  # 실제 수익률이 시뮬레이션 중 몇 %tile
    p_value: float  # 통계적 유의성
    is_significant: bool  # p < 0.05
```

---

## 4. Tier 3: 프로덕션급 고도화

### 4.1 하이브리드 LLM + RL 레이어

```
신규 모듈: src/rl/
```

**근거**: Hybrid LLM+RL = Sharpe 1.57 vs Pure RL 1.35 vs Traditional 0.95
(arXiv:2512.10913, Systematic Review 2025)

**설계**:
```
LLM Layer (현재 ATLAS)     RL Layer (신규)
────────────────────       ──────────────────
시장 분석 + 방향 예측  →   포지션 사이징 최적화
뉴스/센티먼트 해석    →   진입/청산 타이밍
정성적 추론          →   리스크 조절 (동적)

LLM이 "BUY BTCUSDT, confidence 0.8" →
RL이 "optimal weight=0.12, entry_timing=next_dip"
```

```python
# src/rl/position_sizer.py
class RLPositionSizer:
    """강화학습 기반 포지션 사이징.

    State:  [portfolio_state, signal, regime, volatility, indicators]
    Action: [weight_adjustment: -0.1 ~ +0.1]
    Reward: risk_adjusted_return (Sharpe-based)

    LLM의 신호를 받아서 최적 비중을 결정.
    LLM이 BUY 0.2라고 해도 RL이 0.12로 조절 가능.
    """

    def optimize_weight(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        regime: MarketRegime,
        indicators: TechnicalIndicators,
    ) -> float:
        """최적화된 포지션 비중 반환."""
        ...

# src/rl/execution_timer.py
class RLExecutionTimer:
    """강화학습 기반 실행 타이밍 최적화.

    LLM이 BUY라고 해도, 현재가 최적 진입 시점인지 판단.
    - "지금 매수" vs "N분 후 매수" vs "다음 사이클까지 대기"
    """
    ...
```

### 4.2 구조화된 에이전트 간 통신

```
변경: src/agents/multi_agent/graph.py
```

**현재 문제**: 각 단계가 자연어 string만 전달 → "전화기 게임" 효과

**개선 설계**:
```python
@dataclass
class StructuredAnalystReport:
    """자연어 + 구조화 데이터 하이브리드 통신."""

    # 구조화된 핵심 수치 (왜곡 불가)
    top_picks: list[SymbolRating]      # 종목별 평점
    overall_direction: str              # "bullish" | "bearish" | "neutral"
    confidence: float                   # 0.0 ~ 1.0
    risk_flags: list[str]              # 리스크 요인

    # 자연어 상세 분석 (깊이 있는 추론)
    detailed_reasoning: str

    # 메타 데이터
    analyst_type: str                   # "fundamental" | "technical" | ...
    data_sources_used: list[str]        # 참고한 데이터 소스

@dataclass
class SymbolRating:
    symbol: str
    rating: float           # -1.0 ~ +1.0
    target_weight: float    # 추천 비중
    key_factors: list[str]  # 핵심 근거 (3개 이내)
```

### 4.3 결정 귀인 분석 (Decision Attribution)

```
신규 모듈: src/analytics/attribution.py
```

**설계**:
```python
class DecisionAttributor:
    """각 매매 결정이 어떤 데이터에서 비롯되었는지 추적.

    LLM의 reasoning을 파싱하여:
    1. 언급된 데이터 포인트 추출 (가격, 지표, 뉴스)
    2. 실제 snapshot 데이터와 매핑
    3. 각 데이터 포인트의 기여도 추정

    출력 예시:
    Trade: BUY BTCUSDT @ 65,000
    Attribution:
    - Fear&Greed Index = 22 (Extreme Fear) → 40% 기여
    - RSI(14) = 28 (과매도) → 30% 기여
    - 고래 매수 뉴스 → 20% 기여
    - 지지선 근접 → 10% 기여
    """

    def attribute(
        self,
        signal: TradingSignal,
        snapshot: MarketSnapshot,
    ) -> AttributionReport: ...
```

### 4.4 결정론적 리플레이 (Deterministic Replay)

```
신규 모듈: src/simulator/replay.py
```

**설계**:
```python
class SessionRecorder:
    """매 사이클의 모든 입력/출력을 기록하여 재현 가능하게 함.

    기록 항목:
    - MarketSnapshot (전체)
    - 각 LLM의 raw prompt + raw response
    - 파싱된 TradingSignal
    - 리스크 엔진 판단
    - 실행된 Trade
    - 포트폴리오 상태 변화

    AI-Trader (arXiv:2512.10971) 방법론 참고.
    """

    def record_cycle(self, cycle_data: CycleRecord) -> None: ...
    def replay_cycle(self, cycle_id: str) -> CycleRecord: ...
    def replay_session(self, session_id: str) -> list[CycleRecord]: ...

@dataclass
class CycleRecord:
    cycle_id: str
    timestamp: datetime
    snapshots: dict[Market, MarketSnapshot]
    llm_calls: list[LLMCallRecord]  # prompt + response + tokens
    signals: list[TradingSignal]
    risk_decisions: list[RiskDecision]
    trades: list[Trade]
    portfolio_states_before: list[PortfolioState]
    portfolio_states_after: list[PortfolioState]
```

---

## 5. Tier 4: 차세대 로드맵

### 5.1 그래프 뉴럴 네트워크 (자산 관계 모델링)
Nature Scientific Reports (2025) — 종목 간 상관관계를 Graph Attention Network로
모델링. LLM이 놓치는 inter-asset dynamics 포착.

### 5.2 Agent-Based Modeling (ABM)
다수의 AI 에이전트가 동시에 거래할 때 발생하는 emergent behavior 리스크 모델링.
(MixFlow 2025 리서치)

### 5.3 대체 데이터 통합
- 위성 이미지 (원유 저장량, 소매 주차장)
- 신용카드 거래 데이터
- 웹 트래픽 / 앱 다운로드
- 지오로케이션 / 유동 인구

### 5.4 MCP (Model Context Protocol) 도구 체인
Anthropic MCP를 활용한 표준화된 도구 통합. 각 에이전트가 직접 API 호출
가능한 tool-use 기반 아키텍처로 전환.

### 5.5 신뢰도 캘리브레이션 분석
에이전트가 "confidence 0.8"이라고 했을 때, 실제 80% 확률로 맞는지 분석.
캘리브레이션 커브 생성 및 보정.

---

## 6. 신규 디렉터리 구조

```
src/
├── core/              # (기존 유지)
├── data/
│   ├── adapters/
│   │   ├── (기존 6개)
│   │   ├── onchain_adapter.py    # [NEW] 온체인 분석
│   │   └── social_adapter.py     # [NEW] 소셜 센티먼트
│   ├── indicators.py             # [NEW] 기술적 지표 엔진
│   └── (기존 normalizer, cache, scheduler, db)
├── llm/
│   ├── hallucination_detector.py # [NEW] 환각 감지
│   └── (기존 유지)
├── agents/
│   ├── reflection.py             # [NEW] 자기 반성
│   ├── consensus.py              # [NEW] 멀티모델 합의
│   └── (기존 유지)
├── simulator/
│   ├── baselines.py              # [NEW] 다중 베이스라인
│   ├── risk_engine.py            # [NEW] 하드코딩 리스크 엔진
│   ├── replay.py                 # [NEW] 결정론적 리플레이
│   └── (기존 유지)
├── analytics/
│   ├── regime_detector.py        # [NEW] 레짐 감지
│   ├── walk_forward.py           # [NEW] Walk-Forward + Monte Carlo
│   ├── attribution.py            # [NEW] 결정 귀인 분석
│   ├── calibration.py            # [NEW] 신뢰도 캘리브레이션
│   └── (기존 유지)
├── rl/                           # [NEW] 강화학습 레이어
│   ├── __init__.py
│   ├── position_sizer.py
│   ├── execution_timer.py
│   └── CLAUDE.md
└── dashboard/
    └── pages/
        ├── (기존 6개)
        ├── risk_dashboard.py     # [NEW] 리스크 대시보드
        ├── regime_analysis.py    # [NEW] 레짐 분석 페이지
        └── replay_viewer.py      # [NEW] 리플레이 뷰어
```

신규 파일: **약 20개**
수정 파일: **약 15개** (MarketSnapshot 확장, 프롬프트에 지표/레짐 주입 등)

---

## 7. 구현 우선순위

### Phase 1: 신뢰성 확보 (1-2주)
| # | 항목 | 임팩트 | 난이도 |
|---|------|--------|--------|
| 1 | 기술적 지표 엔진 + 프롬프트 주입 | ★★★★★ | ★★☆ |
| 2 | 다중 베이스라인 | ★★★★★ | ★★☆ |
| 3 | 환각 감지 레이어 | ★★★★☆ | ★★★ |
| 4 | 하드코딩 리스크 엔진 | ★★★★★ | ★★★ |
| 5 | 레짐 감지 + 레짐별 평가 | ★★★★☆ | ★★★ |

### Phase 2: 차별화 (2-3주)
| # | 항목 | 임팩트 | 난이도 |
|---|------|--------|--------|
| 6 | 자기 반성 메커니즘 | ★★★★☆ | ★★☆ |
| 7 | 멀티모델 합의 엔진 | ★★★★☆ | ★★☆ |
| 8 | 온체인 분석 어댑터 | ★★★☆☆ | ★★★ |
| 9 | 소셜 센티먼트 파이프라인 | ★★★☆☆ | ★★★ |
| 10 | Walk-Forward + Monte Carlo | ★★★★☆ | ★★★ |

### Phase 3: 고도화 (3-4주)
| # | 항목 | 임팩트 | 난이도 |
|---|------|--------|--------|
| 11 | RL 포지션 사이징 레이어 | ★★★★★ | ★★★★★ |
| 12 | 구조화된 에이전트 통신 | ★★★☆☆ | ★★★ |
| 13 | 결정 귀인 분석 | ★★★☆☆ | ★★★★ |
| 14 | 결정론적 리플레이 | ★★★☆☆ | ★★★ |
| 15 | 신뢰도 캘리브레이션 | ★★★☆☆ | ★★★ |
| 16 | 대시보드 3페이지 추가 | ★★☆☆☆ | ★★☆ |

---

## 8. 참고 문헌

### 벤치마크 프레임워크
- StockBench: Can LLM Agents Trade Stocks Profitably? (arXiv:2510.02209)
- AI-Trader: Fully Autonomous Trading Environment (arXiv:2512.10971)
- FinRL: Deep RL Framework for Trading (GitHub: AI4Finance-Foundation/FinRL)

### 멀티 에이전트 트레이딩
- TradingAgents v0.2: Multi-Agent LLM on LangGraph (GitHub: TauricResearch)
- TradingGroup: Self-Reflection + Data Synthesis (arXiv:2508.17565)
- FINCON: NeurIPS 2024 Multi-Agent Financial Decision Making
- QuantAgent: Multi-Agent LLMs for HFT (arXiv:2509.09995)

### 하이브리드 LLM+RL
- RL in Financial Decision Making: Systematic Review (arXiv:2512.10913)
- PrimoGPT + PrimoRL: LLM + DRL Framework (MDPI 2025)
- FinRL-DeepSeek: LLM Risk Signal + RL Trading (FinRL Contest 2025)

### LLM 한계
- Agent Trading Arena: Numerical Understanding (arXiv:2502.17967)
- LLM Agents Do Not Replicate Human Traders (arXiv:2502.15800)
- LLM Agents for Investment Management (SSRN:5447274)

### 리스크 & 규제
- CFTC AI Advisory (December 2024)
- SEC Rule 15c3-5: Algorithm Risk Management
- EU AI Act: High-Risk Financial AI (2025-2026)
- GAO Report: AI in Financial Services (2025)

### 대체 데이터
- Alternative Data 2025 (Coalition Greenwich): $273B market by 2032
- Graph Attention Multi-Agent RL (Nature Scientific Reports, 2025)
