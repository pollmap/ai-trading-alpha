"use client";

import MetricCard from "@/components/MetricCard";

const MODELS = ["DeepSeek", "Gemini", "Claude", "GPT-4o-mini"];
const ARCHITECTURES = ["Single", "Multi"];

const MOCK_PORTFOLIO_DATA = [
  { model: "DeepSeek", arch: "Single", return: 8.2, sharpe: 1.42, maxDD: -5.1, trades: 47, cost: 12.50 },
  { model: "DeepSeek", arch: "Multi", return: 11.3, sharpe: 1.87, maxDD: -4.2, trades: 38, cost: 45.20 },
  { model: "Gemini", arch: "Single", return: 6.8, sharpe: 1.15, maxDD: -6.3, trades: 52, cost: 8.30 },
  { model: "Gemini", arch: "Multi", return: 9.5, sharpe: 1.55, maxDD: -5.8, trades: 41, cost: 32.10 },
  { model: "Claude", arch: "Single", return: 9.1, sharpe: 1.65, maxDD: -4.5, trades: 44, cost: 15.80 },
  { model: "Claude", arch: "Multi", return: 12.7, sharpe: 2.01, maxDD: -3.8, trades: 35, cost: 58.40 },
  { model: "GPT-4o-mini", arch: "Single", return: 5.9, sharpe: 0.98, maxDD: -7.1, trades: 56, cost: 6.20 },
  { model: "GPT-4o-mini", arch: "Multi", return: 8.1, sharpe: 1.38, maxDD: -6.0, trades: 43, cost: 22.80 },
  { model: "Buy & Hold", arch: "-", return: 4.2, sharpe: 0.72, maxDD: -8.5, trades: 1, cost: 0 },
];

export default function OverviewPage() {
  const bestReturn = MOCK_PORTFOLIO_DATA.reduce((a, b) => a.return > b.return ? a : b);
  const avgReturn = MOCK_PORTFOLIO_DATA.reduce((sum, d) => sum + d.return, 0) / MOCK_PORTFOLIO_DATA.length;
  const totalCost = MOCK_PORTFOLIO_DATA.reduce((sum, d) => sum + d.cost, 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Overview</h1>
        <p className="text-atlas-muted mt-1">
          AI Trading Benchmark — 4 LLMs × 2 Architectures across 3 Markets
        </p>
      </div>

      {/* Top Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Best Performer"
          value={`${bestReturn.model} / ${bestReturn.arch}`}
          delta={`+${bestReturn.return}%`}
          deltaType="positive"
        />
        <MetricCard
          label="Average Return"
          value={`${avgReturn.toFixed(1)}%`}
          delta="All portfolios"
        />
        <MetricCard
          label="Total API Cost"
          value={`$${totalCost.toFixed(2)}`}
          delta="Across all models"
        />
        <MetricCard
          label="Active Portfolios"
          value="27"
          delta="9 per market × 3 markets"
        />
      </div>

      {/* Performance Table */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">Portfolio Performance Summary</h2>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Architecture</th>
                <th>Return (%)</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Trades</th>
                <th>API Cost ($)</th>
                <th>CAA</th>
              </tr>
            </thead>
            <tbody>
              {MOCK_PORTFOLIO_DATA.map((row, i) => {
                const caa = row.cost > 0
                  ? ((row.return - 4.2) / row.cost).toFixed(2)
                  : "N/A";
                return (
                  <tr key={i} className="hover:bg-atlas-border/30 transition-colors">
                    <td className="font-medium text-white">{row.model}</td>
                    <td>{row.arch}</td>
                    <td className={row.return > 0 ? "text-atlas-green" : "text-atlas-red"}>
                      {row.return > 0 ? "+" : ""}{row.return}%
                    </td>
                    <td>{row.sharpe.toFixed(2)}</td>
                    <td className="text-atlas-red">{row.maxDD}%</td>
                    <td>{row.trades}</td>
                    <td>${row.cost.toFixed(2)}</td>
                    <td className="font-medium">{caa}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Market Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
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
              }`}>
                {m.status}
              </span>
            </div>
            <p className="text-sm text-atlas-muted mt-2">Hours: {m.time}</p>
            <p className="text-sm text-atlas-muted">Symbols: {m.symbols}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
