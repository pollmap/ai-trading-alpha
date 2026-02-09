"use client";

import MetricCard from "@/components/MetricCard";
import ExportButton from "@/components/ExportButton";
import { useAutoRefresh } from "@/lib/useAutoRefresh";
import { useI18n } from "@/lib/i18n";
import type { PortfoliosResponse } from "@/lib/api";

const FALLBACK: PortfoliosResponse = {
  portfolios: [
    { model: "deepseek", architecture: "single", market: "CRYPTO", total_value: 108200, initial_capital: 100000, cash: 35200, return_pct: 8.2, sharpe_ratio: 1.42, max_drawdown: -5.1, total_trades: 47, win_rate: 0.58, api_cost: 12.50 },
    { model: "deepseek", architecture: "multi", market: "CRYPTO", total_value: 111300, initial_capital: 100000, cash: 28100, return_pct: 11.3, sharpe_ratio: 1.87, max_drawdown: -4.2, total_trades: 38, win_rate: 0.62, api_cost: 45.20 },
    { model: "gemini", architecture: "single", market: "CRYPTO", total_value: 106800, initial_capital: 100000, cash: 42300, return_pct: 6.8, sharpe_ratio: 1.15, max_drawdown: -6.3, total_trades: 52, win_rate: 0.55, api_cost: 8.30 },
    { model: "gemini", architecture: "multi", market: "CRYPTO", total_value: 109500, initial_capital: 100000, cash: 31200, return_pct: 9.5, sharpe_ratio: 1.55, max_drawdown: -5.8, total_trades: 41, win_rate: 0.57, api_cost: 32.10 },
    { model: "claude", architecture: "single", market: "CRYPTO", total_value: 109100, initial_capital: 100000, cash: 38500, return_pct: 9.1, sharpe_ratio: 1.65, max_drawdown: -4.5, total_trades: 44, win_rate: 0.62, api_cost: 15.80 },
    { model: "claude", architecture: "multi", market: "CRYPTO", total_value: 112700, initial_capital: 100000, cash: 22800, return_pct: 12.7, sharpe_ratio: 2.01, max_drawdown: -3.8, total_trades: 35, win_rate: 0.66, api_cost: 58.40 },
    { model: "gpt-4o-mini", architecture: "single", market: "CRYPTO", total_value: 105900, initial_capital: 100000, cash: 48200, return_pct: 5.9, sharpe_ratio: 0.98, max_drawdown: -7.1, total_trades: 56, win_rate: 0.52, api_cost: 6.20 },
    { model: "gpt-4o-mini", architecture: "multi", market: "CRYPTO", total_value: 108100, initial_capital: 100000, cash: 33400, return_pct: 8.1, sharpe_ratio: 1.38, max_drawdown: -6.0, total_trades: 43, win_rate: 0.56, api_cost: 22.80 },
    { model: "buy_and_hold", architecture: "baseline", market: "CRYPTO", total_value: 104200, initial_capital: 100000, cash: 0, return_pct: 4.2, sharpe_ratio: 0.72, max_drawdown: -8.5, total_trades: 1, win_rate: 1.0, api_cost: 0 },
  ],
  timestamp: new Date().toISOString(),
  market: "CRYPTO",
};

const MODEL_NAMES: Record<string, string> = {
  deepseek: "DeepSeek", gemini: "Gemini", claude: "Claude",
  "gpt-4o-mini": "GPT-4o-mini", buy_and_hold: "Buy & Hold",
};
const ARCH_NAMES: Record<string, string> = { single: "Single", multi: "Multi", baseline: "-" };
const BH_RETURN = 4.2;

export default function OverviewPage() {
  const { t } = useI18n();
  const { data, loading, lastUpdated, refreshing, refresh } = useAutoRefresh<PortfoliosResponse>({
    path: "/api/portfolios",
    fallback: FALLBACK,
  });
  const portfolios = data.portfolios;

  const bestReturn = portfolios.reduce((a, b) => a.return_pct > b.return_pct ? a : b);
  const avgReturn = portfolios.reduce((sum, d) => sum + d.return_pct, 0) / portfolios.length;
  const totalCost = portfolios.reduce((sum, d) => sum + d.api_cost, 0);

  const exportData = portfolios.map((p) => ({
    model: MODEL_NAMES[p.model] || p.model,
    architecture: ARCH_NAMES[p.architecture] || p.architecture,
    return_pct: p.return_pct,
    sharpe_ratio: p.sharpe_ratio,
    max_drawdown: p.max_drawdown,
    total_trades: p.total_trades,
    api_cost: p.api_cost,
    caa: p.api_cost > 0 ? Number(((p.return_pct - BH_RETURN) / p.api_cost).toFixed(3)) : 0,
  }));

  return (
    <div className="space-y-6">
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
        <MetricCard label={t("bestPerformer")} value={`${MODEL_NAMES[bestReturn.model] || bestReturn.model} / ${ARCH_NAMES[bestReturn.architecture] || bestReturn.architecture}`} delta={`+${bestReturn.return_pct}%`} deltaType="positive" />
        <MetricCard label={t("avgReturn")} value={`${avgReturn.toFixed(1)}%`} delta={t("allPortfolios")} />
        <MetricCard label={t("totalApiCost")} value={`$${totalCost.toFixed(2)}`} delta={t("acrossModels")} />
        <MetricCard label={t("activePortfolios")} value="27" delta={t("perMarket")} />
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
                <th>{t("model")}</th><th>{t("arch")}</th><th>{t("returnPct")}</th>
                <th>{t("sharpe")}</th><th>{t("maxDD")}</th>
                <th className="hidden sm:table-cell">{t("trades")}</th>
                <th className="hidden sm:table-cell">{t("cost")}</th><th>{t("caa")}</th>
              </tr>
            </thead>
            <tbody>
              {portfolios.map((row, i) => {
                const caa = row.api_cost > 0 ? ((row.return_pct - BH_RETURN) / row.api_cost).toFixed(2) : "N/A";
                return (
                  <tr key={i} className="hover:bg-atlas-border/30 transition-colors">
                    <td className="font-medium text-white">{MODEL_NAMES[row.model] || row.model}</td>
                    <td>{ARCH_NAMES[row.architecture] || row.architecture}</td>
                    <td className={row.return_pct > 0 ? "text-atlas-green" : "text-atlas-red"}>
                      {row.return_pct > 0 ? "+" : ""}{row.return_pct}%
                    </td>
                    <td>{row.sharpe_ratio.toFixed(2)}</td>
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

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
        {[
          { market: "KRX", status: "Closed", time: "09:00-15:30 KST", symbols: 10 },
          { market: "US", status: "Pre-Market", time: "09:30-16:00 EST", symbols: 10 },
          { market: "CRYPTO", status: "Active", time: "24/7", symbols: 10 },
        ].map((m) => (
          <div key={m.market} className="metric-card">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">{m.market}</h3>
              <span className={`text-xs px-2 py-1 rounded-full ${
                m.status === "Active" ? "bg-green-900/30 text-green-400" :
                m.status === "Pre-Market" ? "bg-yellow-900/30 text-yellow-400" :
                "bg-gray-900/30 text-gray-400"
              }`}>{m.status}</span>
            </div>
            <p className="text-sm text-atlas-muted mt-2">Hours: {m.time}</p>
            <p className="text-sm text-atlas-muted">Symbols: {m.symbols}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
