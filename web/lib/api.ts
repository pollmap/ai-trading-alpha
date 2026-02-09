const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

export async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    next: { revalidate: 30 },
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

export interface Portfolio {
  model: string;
  architecture: string;
  market: string;
  total_value: number;
  initial_capital: number;
  cash: number;
  return_pct: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_trades: number;
  win_rate: number;
  api_cost: number;
}

export interface PortfoliosResponse {
  portfolios: Portfolio[];
  timestamp: string;
  market: string;
}

export interface RiskCheck {
  name: string;
  passed: boolean;
  value: number;
  limit: number;
}

export interface Position {
  symbol: string;
  weight: number;
  value: number;
}

export interface RiskResponse {
  portfolio_var_95: number;
  max_drawdown: number;
  current_drawdown: number;
  daily_loss: number;
  volatility_regime: string;
  checks: RiskCheck[];
  positions: Position[];
  timestamp: string;
}

export interface RegimeInfo {
  regime: string;
  description: string;
}

export interface RegimeTimeline {
  week: number;
  regime: string;
}

export interface RegimeResponse {
  regimes: Record<string, RegimeInfo>;
  timeline: RegimeTimeline[];
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
  version: string;
}
