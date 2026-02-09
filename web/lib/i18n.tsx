"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

type Locale = "ko" | "en";

const translations = {
  ko: {
    // Sidebar
    overview: "개요",
    modelComparison: "모델 비교",
    systemStatus: "시스템 상태",
    riskDashboard: "리스크",
    regimeAnalysis: "레짐 분석",
    replayViewer: "리플레이",
    systemOnline: "시스템 온라인",
    // Overview
    overviewTitle: "개요",
    overviewDesc: "AI 트레이딩 벤치마크 -- 4 LLM x 2 아키텍처, 3개 시장",
    bestPerformer: "최고 성과",
    avgReturn: "평균 수익률",
    totalApiCost: "총 API 비용",
    activePortfolios: "활성 포트폴리오",
    allPortfolios: "전체 포트폴리오",
    acrossModels: "전 모델 합계",
    perMarket: "시장당 9개 x 3",
    perfSummary: "포트폴리오 성과 요약",
    loadingApi: "API에서 불러오는 중...",
    model: "모델",
    arch: "아키텍처",
    returnPct: "수익률",
    sharpe: "샤프비율",
    maxDD: "최대 낙폭",
    trades: "거래수",
    cost: "비용",
    caa: "CAA",
    // Model Comparison
    modelCompTitle: "모델 비교",
    modelCompDesc: "전체 LLM 프로바이더 성과 비교",
    avgReturnLabel: "평균 수익률",
    sharpeRatio: "샤프 비율",
    maxDrawdown: "최대 낙폭",
    winRate: "승률",
    costPerTrade: "거래당 비용",
    singleVsMulti: "Single vs Multi 에이전트 아키텍처",
    avgApiCost: "평균 API 비용",
    caaRanking: "비용조정알파(CAA) 순위",
    rank: "순위",
    vsBH: "vs B&H",
    apiCost: "API 비용",
    caaScore: "CAA 점수",
    // System Status
    sysStatusTitle: "시스템 상태",
    sysStatusDesc: "인프라 모니터링 및 API 상태",
    totalApiCalls: "총 API 호출",
    avgLatency: "평균 지연시간",
    errorRate: "에러율",
    last24h: "최근 24시간",
    llmProviderStatus: "LLM 프로바이더 상태",
    provider: "프로바이더",
    status: "상태",
    latency: "지연시간",
    apiCalls: "API 호출수",
    errors: "에러",
    infrastructure: "인프라",
    recentErrors: "최근 에러 (24시간)",
    // Risk
    riskTitle: "리스크 대시보드",
    riskDesc: "실시간 리스크 모니터링 및 포트폴리오 건강도",
    portfolioVar: "포트폴리오 VaR (95%)",
    currentDD: "현재 낙폭",
    dailyLoss: "일간 손실",
    withinLimit: "한도 내",
    riskCheckResults: "리스크 체크 결과 (마지막 사이클)",
    check: "체크",
    currentValue: "현재값",
    limit: "한도",
    headroom: "여유분",
    positionConcentration: "포지션 집중도",
    volatilityRegime: "현재 변동성 레짐",
    riskAlerts: "리스크 알림",
    noAlerts: "활성 알림 없음. 모든 체크 통과 중.",
    equityCurve: "수익률 곡선",
    drawdownChart: "낙폭 추이",
    // Regime
    regimeTitle: "레짐 분석",
    regimeDesc: "시장 레짐 감지 및 성과 분석",
    regimeTimeline: "레짐 타임라인 (US 시장)",
    perfByRegime: "레짐별 모델 성과",
    bestRegime: "최적 레짐",
    transitionProb: "레짐 전환 확률",
    // Replay
    replayTitle: "리플레이 뷰어",
    replayDesc: "기록된 트레이딩 세션 조회",
    selectSession: "세션 선택",
    totalEvents: "총 이벤트",
    snapshots: "스냅샷",
    signals: "시그널",
    eventFilter: "이벤트 필터",
    eventTimeline: "이벤트 타임라인",
    decisionTrail: "의사결정 경로",
    // Export
    exportCSV: "CSV 다운로드",
    exportJSON: "JSON 다운로드",
    // Auto refresh
    autoRefresh: "자동 갱신",
    lastUpdated: "마지막 갱신",
    refreshing: "갱신 중...",
  },
  en: {
    overview: "Overview",
    modelComparison: "Model Comparison",
    systemStatus: "System Status",
    riskDashboard: "Risk Dashboard",
    regimeAnalysis: "Regime Analysis",
    replayViewer: "Replay Viewer",
    systemOnline: "System Online",
    overviewTitle: "Overview",
    overviewDesc: "AI Trading Benchmark -- 4 LLMs x 2 Architectures across 3 Markets",
    bestPerformer: "Best Performer",
    avgReturn: "Average Return",
    totalApiCost: "Total API Cost",
    activePortfolios: "Active Portfolios",
    allPortfolios: "All portfolios",
    acrossModels: "Across all models",
    perMarket: "9 per market x 3",
    perfSummary: "Portfolio Performance Summary",
    loadingApi: "Loading from API...",
    model: "Model",
    arch: "Arch",
    returnPct: "Return",
    sharpe: "Sharpe",
    maxDD: "Max DD",
    trades: "Trades",
    cost: "Cost",
    caa: "CAA",
    modelCompTitle: "Model Comparison",
    modelCompDesc: "Head-to-head performance comparison across all LLM providers",
    avgReturnLabel: "Avg Return",
    sharpeRatio: "Sharpe Ratio",
    maxDrawdown: "Max Drawdown",
    winRate: "Win Rate",
    costPerTrade: "Cost/Trade",
    singleVsMulti: "Single vs Multi-Agent Architecture",
    avgApiCost: "Avg API Cost",
    caaRanking: "Cost-Adjusted Alpha (CAA) Ranking",
    rank: "Rank",
    vsBH: "vs B&H",
    apiCost: "API Cost",
    caaScore: "CAA Score",
    sysStatusTitle: "System Status",
    sysStatusDesc: "Infrastructure monitoring and API health",
    totalApiCalls: "Total API Calls",
    avgLatency: "Avg Latency",
    errorRate: "Error Rate",
    last24h: "Last 24h",
    llmProviderStatus: "LLM Provider Status",
    provider: "Provider",
    status: "Status",
    latency: "Latency",
    apiCalls: "API Calls",
    errors: "Errors",
    infrastructure: "Infrastructure",
    recentErrors: "Recent Errors (Last 24h)",
    riskTitle: "Risk Dashboard",
    riskDesc: "Real-time risk monitoring and portfolio health",
    portfolioVar: "Portfolio VaR (95%)",
    currentDD: "Current Drawdown",
    dailyLoss: "Daily Loss",
    withinLimit: "Within limit",
    riskCheckResults: "Risk Check Results (Last Cycle)",
    check: "Check",
    currentValue: "Current Value",
    limit: "Limit",
    headroom: "Headroom",
    positionConcentration: "Position Concentration",
    volatilityRegime: "Current Volatility Regime",
    riskAlerts: "Risk Alerts",
    noAlerts: "No active alerts. All checks passing.",
    equityCurve: "Equity Curve",
    drawdownChart: "Drawdown Chart",
    regimeTitle: "Regime Analysis",
    regimeDesc: "Market regime detection and performance breakdown",
    regimeTimeline: "Regime Timeline (US Market)",
    perfByRegime: "Model Performance by Regime",
    bestRegime: "Best Regime",
    transitionProb: "Regime Transition Probabilities",
    replayTitle: "Replay Viewer",
    replayDesc: "Browse and inspect recorded trading sessions",
    selectSession: "Select Session",
    totalEvents: "Total Events",
    snapshots: "Snapshots",
    signals: "Signals",
    eventFilter: "Event Filter",
    eventTimeline: "Event Timeline",
    decisionTrail: "Decision Trail",
    exportCSV: "Export CSV",
    exportJSON: "Export JSON",
    autoRefresh: "Auto Refresh",
    lastUpdated: "Last Updated",
    refreshing: "Refreshing...",
  },
} as const;

type TranslationKey = keyof typeof translations.ko;

interface I18nContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: TranslationKey) => string;
}

const I18nContext = createContext<I18nContextType>({
  locale: "ko",
  setLocale: () => {},
  t: (key) => key,
});

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("ko");

  useEffect(() => {
    const saved = localStorage.getItem("atlas-locale") as Locale | null;
    if (saved === "ko" || saved === "en") setLocaleState(saved);
  }, []);

  const setLocale = (l: Locale) => {
    setLocaleState(l);
    localStorage.setItem("atlas-locale", l);
  };

  const t = (key: TranslationKey): string => {
    return translations[locale][key] || key;
  };

  return (
    <I18nContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  return useContext(I18nContext);
}
