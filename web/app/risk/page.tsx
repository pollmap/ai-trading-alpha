"use client";

import MetricCard from "@/components/MetricCard";

const RISK_CHECKS = [
  { check: "Position Limit", status: "PASS", value: "22.0%", limit: "30.0%" },
  { check: "Cash Reserve", status: "PASS", value: "35.2%", limit: "20.0%" },
  { check: "Confidence Threshold", status: "PASS", value: "0.75", limit: "0.30" },
  { check: "Drawdown Circuit Breaker", status: "PASS", value: "3.1%", limit: "15.0%" },
  { check: "Daily Loss Limit", status: "PASS", value: "0.4%", limit: "5.0%" },
];

const POSITIONS = [
  { symbol: "BTCUSDT", weight: 22, limit: 30, value: "$22,000" },
  { symbol: "ETHUSDT", weight: 18, limit: 30, value: "$18,000" },
  { symbol: "AAPL", weight: 15, limit: 30, value: "$15,000" },
  { symbol: "005930", weight: 10, limit: 30, value: "$10,000" },
];

export default function RiskDashboardPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Risk Dashboard</h1>
        <p className="text-atlas-muted mt-1">Real-time risk monitoring and portfolio health</p>
      </div>

      {/* Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard label="Portfolio VaR (95%)" value="2.3%" delta="-0.1%" deltaType="positive" />
        <MetricCard label="Max Drawdown" value="8.2%" delta="+1.5%" deltaType="negative" />
        <MetricCard label="Current Drawdown" value="3.1%" delta="-0.8%" deltaType="positive" />
        <MetricCard label="Daily Loss" value="0.4%" delta="Within limit" deltaType="positive" />
      </div>

      {/* Risk Checks Table */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">Risk Check Results (Last Cycle)</h2>
        <table className="data-table">
          <thead>
            <tr><th>Check</th><th>Status</th><th>Current Value</th><th>Limit</th><th>Headroom</th></tr>
          </thead>
          <tbody>
            {RISK_CHECKS.map((r) => {
              const current = parseFloat(r.value);
              const limit = parseFloat(r.limit);
              const headroom = Math.abs(limit - current);
              return (
                <tr key={r.check} className="hover:bg-atlas-border/30">
                  <td className="font-medium text-white">{r.check}</td>
                  <td><span className="badge-pass">{r.status}</span></td>
                  <td>{r.value}</td>
                  <td>{r.limit}</td>
                  <td className="text-atlas-green">{headroom.toFixed(1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Position Concentration */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">Position Concentration</h2>
        <div className="space-y-4">
          {POSITIONS.map((pos) => (
            <div key={pos.symbol} className="flex items-center gap-4">
              <span className="text-sm font-medium text-white w-24">{pos.symbol}</span>
              <div className="flex-1">
                <div className="w-full bg-atlas-border rounded-full h-3 relative">
                  <div
                    className="bg-atlas-accent rounded-full h-3 transition-all"
                    style={{ width: `${(pos.weight / pos.limit) * 100}%` }}
                  />
                  <div
                    className="absolute top-0 h-3 border-r-2 border-atlas-yellow"
                    style={{ left: "100%" }}
                  />
                </div>
              </div>
              <span className="text-sm text-atlas-muted w-32 text-right">
                {pos.weight}% / {pos.limit}%
              </span>
              <span className="text-sm text-white w-24 text-right">{pos.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Volatility Regime */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="metric-card">
          <h3 className="text-sm text-atlas-muted">Current Volatility Regime</h3>
          <p className="text-2xl font-bold text-atlas-green mt-1">NORMAL</p>
          <p className="text-sm text-atlas-muted mt-2">Annualised Vol: 18.3%</p>
          <p className="text-sm text-atlas-muted">Duration: 5 days</p>
        </div>
        <div className="metric-card">
          <h3 className="text-sm text-atlas-muted">Risk Alerts</h3>
          <div className="flex items-center gap-2 mt-3">
            <div className="w-3 h-3 rounded-full bg-atlas-green" />
            <p className="text-sm text-atlas-green">No active alerts. All checks passing.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
