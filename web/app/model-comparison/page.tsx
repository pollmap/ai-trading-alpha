"use client";

import MetricCard from "@/components/MetricCard";

const COMPARISON_DATA = {
  metrics: [
    { model: "DeepSeek", return: 9.75, sharpe: 1.645, maxDD: 4.65, winRate: 58, costPerTrade: 0.27 },
    { model: "Gemini", return: 8.15, sharpe: 1.350, maxDD: 6.05, winRate: 55, costPerTrade: 0.20 },
    { model: "Claude", return: 10.90, sharpe: 1.830, maxDD: 4.15, winRate: 62, costPerTrade: 0.37 },
    { model: "GPT-4o-mini", return: 7.00, sharpe: 1.180, maxDD: 6.55, winRate: 52, costPerTrade: 0.15 },
  ],
  archComparison: [
    { arch: "Single Agent", avgReturn: 7.5, avgSharpe: 1.30, avgCost: 10.7 },
    { arch: "Multi Agent", avgReturn: 10.4, avgSharpe: 1.70, avgCost: 39.6 },
  ],
};

export default function ModelComparisonPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Model Comparison</h1>
        <p className="text-atlas-muted mt-1">
          Head-to-head performance comparison across all LLM providers
        </p>
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {COMPARISON_DATA.metrics.map((m) => (
          <div key={m.model} className="metric-card space-y-3">
            <h3 className="text-lg font-semibold text-white">{m.model}</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-atlas-muted">Avg Return</span>
                <span className="text-atlas-green">+{m.return}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-atlas-muted">Sharpe Ratio</span>
                <span className="text-white">{m.sharpe.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-atlas-muted">Max Drawdown</span>
                <span className="text-atlas-red">-{m.maxDD}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-atlas-muted">Win Rate</span>
                <span className="text-white">{m.winRate}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-atlas-muted">Cost/Trade</span>
                <span className="text-atlas-muted">${m.costPerTrade}</span>
              </div>
            </div>
            {/* Performance bar */}
            <div className="w-full bg-atlas-border rounded-full h-2">
              <div
                className="bg-atlas-accent rounded-full h-2 transition-all"
                style={{ width: `${(m.return / 12) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Architecture Comparison */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">Single vs Multi-Agent Architecture</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {COMPARISON_DATA.archComparison.map((arch) => (
            <div key={arch.arch} className="bg-atlas-bg rounded-lg p-4 border border-atlas-border">
              <h3 className="text-md font-medium text-white mb-3">{arch.arch}</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-atlas-muted">Avg Return</span>
                  <span className="text-atlas-green">+{arch.avgReturn}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-atlas-muted">Avg Sharpe</span>
                  <span className="text-white">{arch.avgSharpe.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-atlas-muted">Avg API Cost</span>
                  <span className="text-atlas-muted">${arch.avgCost.toFixed(2)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detailed Comparison Table */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">Cost-Adjusted Alpha (CAA) Ranking</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th>Return (%)</th>
              <th>vs B&H (%)</th>
              <th>API Cost ($)</th>
              <th>CAA Score</th>
            </tr>
          </thead>
          <tbody>
            {COMPARISON_DATA.metrics
              .map((m) => ({
                ...m,
                alpha: m.return - 4.2,
                totalCost: m.costPerTrade * 45,
                caa: (m.return - 4.2) / (m.costPerTrade * 45),
              }))
              .sort((a, b) => b.caa - a.caa)
              .map((m, i) => (
                <tr key={m.model} className="hover:bg-atlas-border/30">
                  <td className="font-bold text-atlas-accent">#{i + 1}</td>
                  <td className="font-medium text-white">{m.model}</td>
                  <td className="text-atlas-green">+{m.return}%</td>
                  <td className="text-atlas-green">+{m.alpha.toFixed(1)}%</td>
                  <td>${m.totalCost.toFixed(2)}</td>
                  <td className="font-bold text-white">{m.caa.toFixed(3)}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
