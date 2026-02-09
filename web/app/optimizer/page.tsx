"use client";

import { useState } from "react";
import { useI18n } from "@/lib/i18n";
import ExportButton from "@/components/ExportButton";

const OPTIMIZATION_METHODS = [
  { id: "mean_variance", color: "blue" },
  { id: "risk_parity", color: "green" },
  { id: "black_litterman", color: "purple" },
  { id: "min_variance", color: "yellow" },
  { id: "max_sharpe", color: "red" },
  { id: "equal_weight", color: "gray" },
];

const MOCK_ASSETS = [
  { symbol: "AAPL", market: "US", weight: 0.18, expectedReturn: 0.12, vol: 0.22 },
  { symbol: "7203.T", market: "JPX", weight: 0.12, expectedReturn: 0.08, vol: 0.18 },
  { symbol: "MC.PA", market: "EURONEXT", weight: 0.15, expectedReturn: 0.15, vol: 0.25 },
  { symbol: "0700.HK", market: "HKEX", weight: 0.10, expectedReturn: 0.10, vol: 0.30 },
  { symbol: "SHEL.L", market: "LSE", weight: 0.10, expectedReturn: 0.06, vol: 0.20 },
  { symbol: "GC=F", market: "COMMODITIES", weight: 0.15, expectedReturn: 0.05, vol: 0.15 },
  { symbol: "TLT", market: "BOND", weight: 0.12, expectedReturn: 0.04, vol: 0.12 },
  { symbol: "BTC-USD", market: "CRYPTO", weight: 0.08, expectedReturn: 0.25, vol: 0.60 },
];

const MOCK_RESULT = {
  method: "max_sharpe",
  expectedReturn: 0.1152,
  expectedVolatility: 0.1845,
  sharpeRatio: 1.42,
  diversificationRatio: 1.85,
};

export default function OptimizerPage() {
  const { t } = useI18n();
  const [selectedMethod, setSelectedMethod] = useState("max_sharpe");

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold">{t("optTitle")}</h1>
          <p className="text-gray-400 mt-1">{t("optDesc")}</p>
        </div>
        <ExportButton data={MOCK_ASSETS} filename="portfolio_optimization" />
      </div>

      {/* Method Selection */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
        {OPTIMIZATION_METHODS.map(m => (
          <button key={m.id} onClick={() => setSelectedMethod(m.id)}
            className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
              selectedMethod === m.id
                ? "bg-blue-600 text-white ring-2 ring-blue-400"
                : "bg-atlas-panel border border-atlas-border text-gray-400 hover:text-white"
            }`}>
            {t(m.id === "mean_variance" ? "meanVariance" :
               m.id === "risk_parity" ? "riskParity" :
               m.id === "black_litterman" ? "blackLitterman" :
               m.id === "min_variance" ? "minVariance" :
               m.id === "max_sharpe" ? "maxSharpeRatio" : "equalWeight")}
          </button>
        ))}
      </div>

      {/* Result Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: t("expectedReturn"), value: `${(MOCK_RESULT.expectedReturn * 100).toFixed(2)}%`, color: "text-green-400" },
          { label: t("expectedVol"), value: `${(MOCK_RESULT.expectedVolatility * 100).toFixed(2)}%`, color: "text-yellow-400" },
          { label: t("sharpe"), value: MOCK_RESULT.sharpeRatio.toFixed(2), color: "text-blue-400" },
          { label: "Diversification", value: `${MOCK_RESULT.diversificationRatio.toFixed(2)}x`, color: "text-purple-400" },
        ].map(item => (
          <div key={item.label} className="bg-atlas-panel border border-atlas-border rounded-xl p-4">
            <div className="text-xs text-gray-400">{item.label}</div>
            <div className={`text-xl font-bold mt-1 ${item.color}`}>{item.value}</div>
          </div>
        ))}
      </div>

      {/* Asset Allocation Table */}
      <div className="bg-atlas-panel border border-atlas-border rounded-xl p-5">
        <h3 className="font-semibold mb-4">{t("targetWeights")}</h3>
        <div className="space-y-3">
          {MOCK_ASSETS.map(asset => (
            <div key={asset.symbol} className="flex items-center gap-4">
              <div className="w-24 text-sm font-medium">{asset.symbol}</div>
              <div className="w-24 text-xs text-gray-400">{asset.market}</div>
              <div className="flex-1">
                <div className="w-full bg-atlas-dark rounded-full h-4 relative">
                  <div
                    className="bg-gradient-to-r from-blue-600 to-blue-400 h-4 rounded-full transition-all duration-500"
                    style={{ width: `${asset.weight * 100 * 2.5}%` }}
                  />
                  <span className="absolute right-2 top-0 text-xs leading-4 text-gray-300">
                    {(asset.weight * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="w-20 text-right text-sm">
                <span className="text-green-400">+{(asset.expectedReturn * 100).toFixed(1)}%</span>
              </div>
              <div className="w-20 text-right text-sm text-gray-400">
                σ {(asset.vol * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Correlation Matrix Heatmap (simplified) */}
      <div className="bg-atlas-panel border border-atlas-border rounded-xl p-5">
        <h3 className="font-semibold mb-4">{t("corrMatrix")}</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr>
                <th className="px-2 py-1"></th>
                {MOCK_ASSETS.slice(0, 6).map(a => (
                  <th key={a.symbol} className="px-2 py-1 text-gray-400">{a.symbol}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MOCK_ASSETS.slice(0, 6).map((row, i) => (
                <tr key={row.symbol}>
                  <td className="px-2 py-1 text-gray-400 font-medium">{row.symbol}</td>
                  {MOCK_ASSETS.slice(0, 6).map((col, j) => {
                    const corr = i === j ? 1.0 : (0.1 + Math.random() * 0.5) * (Math.random() > 0.3 ? 1 : -1);
                    const absCorr = Math.abs(corr);
                    const bg = corr >= 0
                      ? `rgba(34, 197, 94, ${absCorr * 0.5})`
                      : `rgba(239, 68, 68, ${absCorr * 0.5})`;
                    return (
                      <td key={col.symbol} className="px-2 py-1 text-center" style={{ backgroundColor: bg }}>
                        {corr.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
