"use client";

import MetricCard from "@/components/MetricCard";
import ExportButton from "@/components/ExportButton";
import { useI18n } from "@/lib/i18n";
import { useAutoRefresh } from "@/lib/useAutoRefresh";
import { RegimeTimelineChart } from "@/components/PlotlyChart";

const REGIME_COLORS: Record<string, string> = {
  BULL: "bg-green-900/40 text-green-400",
  BEAR: "bg-red-900/40 text-red-400",
  SIDEWAYS: "bg-gray-900/40 text-gray-400",
  HIGH_VOL: "bg-yellow-900/40 text-yellow-400",
  CRASH: "bg-red-900/60 text-red-300",
};

interface RegimeStatus {
  market: string;
  regime: string;
  description: string;
  color: string;
}

interface PerfByRegime {
  model: string;
  bull: number;
  sideways: number;
  bear: number;
}

interface TransitionRow {
  from: string;
  to_bull: string;
  to_side: string;
  to_bear: string;
  to_vol: string;
}

interface RegimeData {
  regime_status: RegimeStatus[];
  perf_by_regime: PerfByRegime[];
  regime_timeline: string[];
  transition_matrix: TransitionRow[];
}

const FALLBACK: RegimeData = {
  regime_status: [
    { market: "KRX", regime: "SIDEWAYS", description: "SMA50 near SMA200, low volatility", color: "text-gray-400" },
    { market: "US", regime: "BULL", description: "SMA50 > SMA200, positive momentum", color: "text-atlas-green" },
    { market: "CRYPTO", regime: "HIGH_VOL", description: "ATR elevated, mixed signals", color: "text-atlas-yellow" },
  ],
  perf_by_regime: [
    { model: "DeepSeek", bull: 8.2, sideways: 1.2, bear: -3.1 },
    { model: "Gemini", bull: 7.5, sideways: 0.8, bear: -2.5 },
    { model: "Claude", bull: 9.1, sideways: 2.1, bear: -1.8 },
    { model: "GPT-4o-mini", bull: 6.8, sideways: -0.5, bear: -4.2 },
  ],
  regime_timeline: ["BULL", "BULL", "SIDEWAYS", "SIDEWAYS", "BEAR", "BEAR", "BEAR", "SIDEWAYS", "BULL", "BULL"],
  transition_matrix: [
    { from: "BULL", to_bull: "70%", to_side: "20%", to_bear: "8%", to_vol: "2%" },
    { from: "SIDEWAYS", to_bull: "15%", to_side: "60%", to_bear: "20%", to_vol: "5%" },
    { from: "BEAR", to_bull: "5%", to_side: "25%", to_bear: "60%", to_vol: "10%" },
    { from: "HIGH_VOL", to_bull: "10%", to_side: "15%", to_bear: "25%", to_vol: "50%" },
  ],
};

export default function RegimeAnalysisPage() {
  const { t } = useI18n();
  const { data, lastUpdated, refreshing, refresh } = useAutoRefresh<RegimeData>({
    path: "/api/regime",
    fallback: FALLBACK,
  });

  const regimeStatus = data.regime_status;
  const perfByRegime = data.perf_by_regime;
  const regimeTimeline = data.regime_timeline;
  const transitionMatrix = data.transition_matrix;

  const timeline = regimeTimeline.map((regime, i) => ({
    week: i + 1,
    regime: regime.toLowerCase(),
  }));

  const perfExportData = perfByRegime.map((row) => {
    const best = row.bull >= row.sideways && row.bull >= row.bear ? "BULL" :
                row.sideways >= row.bear ? "SIDEWAYS" : "BEAR";
    return { model: row.model, bull: `${row.bull}%`, sideways: `${row.sideways}%`, bear: `${row.bear}%`, best_regime: best };
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h1 className="text-3xl font-bold text-white">{t("regimeTitle")}</h1>
          <p className="text-atlas-muted mt-1">{t("regimeDesc")}</p>
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

      {/* Current Regime per Market */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {regimeStatus.map((r) => (
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

      {/* Regime Timeline (Plotly) */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">{t("regimeTimeline")}</h2>
        <RegimeTimelineChart timeline={timeline} />
      </div>

      {/* Performance by Regime */}
      <div className="chart-container">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">{t("perfByRegime")}</h2>
          <ExportButton data={perfExportData} filename="atlas-perf-by-regime" />
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>{t("model")}</th>
              <th>Bull (%)</th>
              <th>Sideways (%)</th>
              <th>Bear (%)</th>
              <th>{t("bestRegime")}</th>
            </tr>
          </thead>
          <tbody>
            {perfByRegime.map((row) => {
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
        <h2 className="text-lg font-semibold text-white mb-4">{t("transitionProb")}</h2>
        <table className="data-table">
          <thead>
            <tr><th>From / To</th><th>BULL</th><th>SIDEWAYS</th><th>BEAR</th><th>HIGH_VOL</th></tr>
          </thead>
          <tbody>
            {transitionMatrix.map((row) => (
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
