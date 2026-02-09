"use client";

import MetricCard from "@/components/MetricCard";
import ExportButton from "@/components/ExportButton";
import { useAutoRefresh } from "@/lib/useAutoRefresh";
import { useI18n } from "@/lib/i18n";
import type { PortfoliosResponse } from "@/lib/api";

// Realistic demo data — includes losses, mixed results, realistic Sharpe ratios
// This is sample data shown when no backend is connected
const FALLBACK: PortfoliosResponse = {
  portfolios: [
    { model: "deepseek", architecture: "single", market: "KRX", total_value: 97800, initial_capital: 100000, cash: 42300, return_pct: -2.2, sharpe_ratio: -0.31, max_drawdown: -8.4, total_trades: 34, win_rate: 0.41, api_cost: 4.80 },
    { model: "deepseek", architecture: "multi", market: "KRX", total_value: 101200, initial_capital: 100000, cash: 28100, return_pct: 1.2, sharpe_ratio: 0.18, max_drawdown: -6.1, total_trades: 28, win_rate: 0.50, api_cost: 18.40 },
    { model: "gemini", architecture: "single", market: "US", total_value: 103400, initial_capital: 100000, cash: 38500, return_pct: 3.4, sharpe_ratio: 0.62, max_drawdown: -5.7, total_trades: 41, win_rate: 0.54, api_cost: 6.20 },
    { model: "gemini", architecture: "multi", market: "US", total_value: 98100, initial_capital: 100000, cash: 31200, return_pct: -1.9, sharpe_ratio: -0.22, max_drawdown: -9.3, total_trades: 36, win_rate: 0.44, api_cost: 24.50 },
    { model: "claude", architecture: "single", market: "CRYPTO", total_value: 105600, initial_capital: 100000, cash: 22800, return_pct: 5.6, sharpe_ratio: 0.89, max_drawdown: -7.2, total_trades: 52, win_rate: 0.56, api_cost: 12.30 },
    { model: "claude", architecture: "multi", market: "CRYPTO", total_value: 102300, initial_capital: 100000, cash: 35400, return_pct: 2.3, sharpe_ratio: 0.34, max_drawdown: -11.5, total_trades: 31, win_rate: 0.48, api_cost: 47.80 },
    { model: "gpt-4o-mini", architecture: "single", market: "KRX", total_value: 96200, initial_capital: 100000, cash: 48200, return_pct: -3.8, sharpe_ratio: -0.55, max_drawdown: -12.1, total_trades: 45, win_rate: 0.38, api_cost: 3.10 },
    { model: "gpt-4o-mini", architecture: "multi", market: "US", total_value: 100800, initial_capital: 100000, cash: 33400, return_pct: 0.8, sharpe_ratio: 0.11, max_drawdown: -7.8, total_trades: 39, win_rate: 0.46, api_cost: 11.60 },
    { model: "buy_and_hold", architecture: "baseline", market: "KRX", total_value: 101500, initial_capital: 100000, cash: 0, return_pct: 1.5, sharpe_ratio: 0.22, max_drawdown: -9.8, total_trades: 1, win_rate: 1.0, api_cost: 0 },
  ],
  timestamp: new Date().toISOString(),
  market: "ALL",
};

const MODEL_NAMES: Record<string, string> = {
  deepseek: "DeepSeek", gemini: "Gemini", claude: "Claude",
  "gpt-4o-mini": "GPT-4o-mini", buy_and_hold: "Buy & Hold",
};
const ARCH_NAMES: Record<string, string> = { single: "Single", multi: "Multi", baseline: "-" };

const ALL_MARKETS = [
  { market: "KRX", status: "Standby", time: "09:00-15:30 KST", symbols: 10, flag: "KR" },
  { market: "US", status: "Standby", time: "09:30-16:00 EST", symbols: 10, flag: "US" },
  { market: "CRYPTO", status: "Standby", time: "24/7", symbols: 10, flag: "BTC" },
  { market: "JPX", status: "Standby", time: "09:00-15:00 JST", symbols: 5, flag: "JP" },
  { market: "SSE", status: "Standby", time: "09:30-15:00 CST", symbols: 5, flag: "CN" },
  { market: "HKEX", status: "Standby", time: "09:30-16:00 HKT", symbols: 5, flag: "HK" },
  { market: "EURONEXT", status: "Standby", time: "09:00-17:30 CET", symbols: 5, flag: "EU" },
  { market: "LSE", status: "Standby", time: "08:00-16:30 GMT", symbols: 5, flag: "GB" },
  { market: "BOND", status: "Standby", time: "Market hours", symbols: 5, flag: "BD" },
  { market: "COMMODITIES", status: "Standby", time: "Market hours", symbols: 5, flag: "CM" },
];

export default function OverviewPage() {
  const { t } = useI18n();
  const { data, loading, lastUpdated, refreshing, refresh } = useAutoRefresh<PortfoliosResponse>({
    path: "/api/portfolios",
    fallback: FALLBACK,
  });
  const portfolios = data.portfolios;
  const isDemo = !lastUpdated || data === FALLBACK;

  const bhRow = portfolios.find((p) => p.model === "buy_and_hold");
  const bhReturn = bhRow ? bhRow.return_pct : 0;
  const agentPortfolios = portfolios.filter((p) => p.model !== "buy_and_hold");
  const bestReturn = agentPortfolios.length > 0
    ? agentPortfolios.reduce((a, b) => a.return_pct > b.return_pct ? a : b)
    : portfolios[0];
  const avgReturn = agentPortfolios.length > 0
    ? agentPortfolios.reduce((sum, d) => sum + d.return_pct, 0) / agentPortfolios.length
    : 0;
  const totalCost = portfolios.reduce((sum, d) => sum + d.api_cost, 0);
  const portfolioCount = portfolios.length;

  const exportData = portfolios.map((p) => ({
    model: MODEL_NAMES[p.model] || p.model,
    architecture: ARCH_NAMES[p.architecture] || p.architecture,
    market: p.market,
    return_pct: p.return_pct,
    sharpe_ratio: p.sharpe_ratio,
    max_drawdown: p.max_drawdown,
    total_trades: p.total_trades,
    api_cost: p.api_cost,
    caa: p.api_cost > 0 ? Number(((p.return_pct - bhReturn) / p.api_cost).toFixed(3)) : 0,
  }));

  return (
    <div className="space-y-6">
      {/* DEMO Banner */}
      {isDemo && (
        <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-xl px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-yellow-400 text-lg">&#9888;</span>
            <div>
              <p className="text-yellow-300 font-semibold text-sm">DEMO MODE — Sample Data</p>
              <p className="text-yellow-400/70 text-xs">Backend not connected. Showing sample data for UI demonstration. Connect to Oracle Cloud backend for live results.</p>
            </div>
          </div>
        </div>
      )}

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white">{t("overviewTitle")}</h1>
          <p className="text-atlas-muted mt-1 text-sm">{t("overviewDesc")}</p>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs text-atlas-muted">
              {refreshing ? t("refreshing") : `${t("lastUpdated")}: ${lastUpdated.toLocaleTimeString()}`}
            </span>
          )}
          <button onClick={refresh} className="px-3 py-1.5 text-xs bg-atlas-accent/20 text-atlas-accent rounded-lg hover:bg-atlas-accent/30 transition-colors">
            {t("autoRefresh")}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <MetricCard
          label={t("bestPerformer")}
          value={`${MODEL_NAMES[bestReturn.model] || bestReturn.model} / ${ARCH_NAMES[bestReturn.architecture] || bestReturn.architecture}`}
          delta={`${bestReturn.return_pct > 0 ? "+" : ""}${bestReturn.return_pct}%`}
          deltaType={bestReturn.return_pct > 0 ? "positive" : "negative"}
        />
        <MetricCard
          label={t("avgReturn")}
          value={`${avgReturn > 0 ? "+" : ""}${avgReturn.toFixed(1)}%`}
          delta={`vs B&H ${bhReturn > 0 ? "+" : ""}${bhReturn}%`}
          deltaType={avgReturn > bhReturn ? "positive" : "negative"}
        />
        <MetricCard label={t("totalApiCost")} value={`$${totalCost.toFixed(2)}`} delta={`${portfolioCount} portfolios`} />
        <MetricCard label={t("activePortfolios")} value={String(portfolioCount)} delta="10 markets available" />
      </div>

      <div className="chart-container">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">{t("perfSummary")}</h2>
          <ExportButton data={exportData} filename="atlas-portfolios" />
        </div>
        {loading && <p className="text-atlas-muted text-sm animate-pulse">{t("loadingApi")}</p>}
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>{t("model")}</th><th>{t("arch")}</th><th>Market</th><th>{t("returnPct")}</th>
                <th>{t("sharpe")}</th><th>{t("maxDD")}</th>
                <th className="hidden sm:table-cell">{t("trades")}</th>
                <th className="hidden sm:table-cell">{t("cost")}</th><th>{t("caa")}</th>
              </tr>
            </thead>
            <tbody>
              {portfolios.map((row, i) => {
                const caa = row.api_cost > 0 ? ((row.return_pct - bhReturn) / row.api_cost).toFixed(2) : "N/A";
                return (
                  <tr key={i} className="hover:bg-atlas-border/30 transition-colors">
                    <td className="font-medium text-white">{MODEL_NAMES[row.model] || row.model}</td>
                    <td>{ARCH_NAMES[row.architecture] || row.architecture}</td>
                    <td className="text-atlas-muted">{row.market}</td>
                    <td className={row.return_pct > 0 ? "text-atlas-green" : "text-atlas-red"}>
                      {row.return_pct > 0 ? "+" : ""}{row.return_pct}%
                    </td>
                    <td className={row.sharpe_ratio >= 0 ? "" : "text-atlas-red"}>{row.sharpe_ratio.toFixed(2)}</td>
                    <td className="text-atlas-red">{row.max_drawdown}%</td>
                    <td className="hidden sm:table-cell">{row.total_trades}</td>
                    <td className="hidden sm:table-cell">${row.api_cost.toFixed(2)}</td>
                    <td className="font-medium">{caa}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 10 Global Markets */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-3">Global Market Coverage</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {ALL_MARKETS.map((m) => (
            <div key={m.market} className="metric-card">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">{m.market}</h3>
                <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800/60 text-gray-400">
                  {m.status}
                </span>
              </div>
              <p className="text-xs text-atlas-muted mt-1.5">{m.time}</p>
              <p className="text-xs text-atlas-muted">{m.symbols} symbols</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
