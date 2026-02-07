"use client";

import MetricCard from "@/components/MetricCard";

const REGIME_STATUS = [
  { market: "KRX", regime: "SIDEWAYS", description: "SMA50 near SMA200, low volatility", color: "text-gray-400" },
  { market: "US", regime: "BULL", description: "SMA50 > SMA200, positive momentum", color: "text-atlas-green" },
  { market: "CRYPTO", regime: "HIGH_VOL", description: "ATR elevated, mixed signals", color: "text-atlas-yellow" },
];

const REGIME_COLORS: Record<string, string> = {
  BULL: "bg-green-900/40 text-green-400",
  BEAR: "bg-red-900/40 text-red-400",
  SIDEWAYS: "bg-gray-900/40 text-gray-400",
  HIGH_VOL: "bg-yellow-900/40 text-yellow-400",
  CRASH: "bg-red-900/60 text-red-300",
};

const PERF_BY_REGIME = [
  { model: "DeepSeek", bull: 8.2, sideways: 1.2, bear: -3.1 },
  { model: "Gemini", bull: 7.5, sideways: 0.8, bear: -2.5 },
  { model: "Claude", bull: 9.1, sideways: 2.1, bear: -1.8 },
  { model: "GPT-4o-mini", bull: 6.8, sideways: -0.5, bear: -4.2 },
];

const REGIME_TIMELINE = ["BULL", "BULL", "SIDEWAYS", "SIDEWAYS", "BEAR", "BEAR", "BEAR", "SIDEWAYS", "BULL", "BULL"];

const TRANSITION_MATRIX = [
  { from: "BULL", to_bull: "70%", to_side: "20%", to_bear: "8%", to_vol: "2%" },
  { from: "SIDEWAYS", to_bull: "15%", to_side: "60%", to_bear: "20%", to_vol: "5%" },
  { from: "BEAR", to_bull: "5%", to_side: "25%", to_bear: "60%", to_vol: "10%" },
  { from: "HIGH_VOL", to_bull: "10%", to_side: "15%", to_bear: "25%", to_vol: "50%" },
];

export default function RegimeAnalysisPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Regime Analysis</h1>
        <p className="text-atlas-muted mt-1">Market regime detection and performance breakdown</p>
      </div>

      {/* Current Regime per Market */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {REGIME_STATUS.map((r) => (
          <div key={r.market} className="metric-card">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">{r.market}</h3>
              <span className={`text-xs px-2.5 py-1 rounded-full ${REGIME_COLORS[r.regime] || "bg-gray-900/40 text-gray-400"}`}>
                {r.regime}
              </span>
            </div>
            <p className="text-sm text-atlas-muted mt-2">{r.description}</p>
          </div>
        ))}
      </div>

      {/* Regime Timeline */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">Regime Timeline (US Market)</h2>
        <div className="flex gap-1">
          {REGIME_TIMELINE.map((regime, i) => (
            <div
              key={i}
              className={`flex-1 h-10 rounded flex items-center justify-center text-xs font-medium ${REGIME_COLORS[regime]}`}
            >
              {regime.slice(0, 4)}
            </div>
          ))}
        </div>
        <div className="flex justify-between mt-1">
          <span className="text-xs text-atlas-muted">Week 1</span>
          <span className="text-xs text-atlas-muted">Week 10</span>
        </div>
      </div>

      {/* Performance by Regime */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">Model Performance by Regime</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Bull (%)</th>
              <th>Sideways (%)</th>
              <th>Bear (%)</th>
              <th>Best Regime</th>
            </tr>
          </thead>
          <tbody>
            {PERF_BY_REGIME.map((row) => {
              const best = row.bull >= row.sideways && row.bull >= row.bear ? "BULL" :
                          row.sideways >= row.bear ? "SIDEWAYS" : "BEAR";
              return (
                <tr key={row.model} className="hover:bg-atlas-border/30">
                  <td className="font-medium text-white">{row.model}</td>
                  <td className="text-atlas-green">+{row.bull}%</td>
                  <td className={row.sideways >= 0 ? "text-atlas-green" : "text-atlas-red"}>
                    {row.sideways >= 0 ? "+" : ""}{row.sideways}%
                  </td>
                  <td className="text-atlas-red">{row.bear}%</td>
                  <td>
                    <span className={`text-xs px-2 py-1 rounded-full ${REGIME_COLORS[best]}`}>
                      {best}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Transition Matrix */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">Regime Transition Probabilities</h2>
        <table className="data-table">
          <thead>
            <tr><th>From / To</th><th>BULL</th><th>SIDEWAYS</th><th>BEAR</th><th>HIGH_VOL</th></tr>
          </thead>
          <tbody>
            {TRANSITION_MATRIX.map((row) => (
              <tr key={row.from} className="hover:bg-atlas-border/30">
                <td className="font-medium text-white">{row.from}</td>
                <td>{row.to_bull}</td>
                <td>{row.to_side}</td>
                <td>{row.to_bear}</td>
                <td>{row.to_vol}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
