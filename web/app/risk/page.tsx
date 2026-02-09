"use client";

import MetricCard from "@/components/MetricCard";
import ExportButton from "@/components/ExportButton";
import { useI18n } from "@/lib/i18n";
import { useAutoRefresh } from "@/lib/useAutoRefresh";
import { EquityCurve, DrawdownChart } from "@/components/PlotlyChart";

interface RiskCheck {
  check: string;
  status: string;
  value: string;
  limit: string;
}

interface Position {
  symbol: string;
  weight: number;
  limit: number;
  value: string;
}

interface RiskData {
  risk_checks: RiskCheck[];
  positions: Position[];
  equity_curve: number[];
  drawdown_series: number[];
}

const FALLBACK: RiskData = {
  risk_checks: [
    { check: "Position Limit", status: "PASS", value: "22.0%", limit: "30.0%" },
    { check: "Cash Reserve", status: "PASS", value: "35.2%", limit: "20.0%" },
    { check: "Confidence Threshold", status: "PASS", value: "0.75", limit: "0.30" },
    { check: "Drawdown Circuit Breaker", status: "PASS", value: "3.1%", limit: "15.0%" },
    { check: "Daily Loss Limit", status: "PASS", value: "0.4%", limit: "5.0%" },
  ],
  positions: [
    { symbol: "BTCUSDT", weight: 22, limit: 30, value: "$22,000" },
    { symbol: "ETHUSDT", weight: 18, limit: 30, value: "$18,000" },
    { symbol: "AAPL", weight: 15, limit: 30, value: "$15,000" },
    { symbol: "005930", weight: 10, limit: 30, value: "$10,000" },
  ],
  equity_curve: [100000, 100800, 101200, 100500, 101800, 103200, 102100, 104500, 105200, 104800, 106100, 107300, 106800, 108200],
  drawdown_series: [0, -0.002, -0.005, -0.012, -0.003, 0, -0.011, 0, -0.002, -0.008, 0, -0.003, -0.008, 0],
};

export default function RiskDashboardPage() {
  const { t } = useI18n();
  const { data, lastUpdated, refreshing, refresh } = useAutoRefresh<RiskData>({
    path: "/api/risk",
    fallback: FALLBACK,
  });

  const riskChecks = data.risk_checks;
  const positions = data.positions;
  const equityCurve = data.equity_curve;
  const drawdownSeries = data.drawdown_series;

  const exportData = riskChecks.map((r) => {
    const current = parseFloat(r.value);
    const limit = parseFloat(r.limit);
    const headroom = Math.abs(limit - current);
    return { check: r.check, status: r.status, current_value: r.value, limit: r.limit, headroom: `${headroom.toFixed(1)}%` };
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
        <div>
          <h1 className="text-3xl font-bold text-white">{t("riskTitle")}</h1>
          <p className="text-atlas-muted mt-1">{t("riskDesc")}</p>
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

      {/* Risk Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard label={t("portfolioVar")} value="2.3%" delta="-0.1%" deltaType="positive" />
        <MetricCard label={t("maxDD")} value="8.2%" delta="+1.5%" deltaType="negative" />
        <MetricCard label={t("currentDD")} value="3.1%" delta="-0.8%" deltaType="positive" />
        <MetricCard label={t("dailyLoss")} value="0.4%" delta={t("withinLimit")} deltaType="positive" />
      </div>

      {/* Equity Curve & Drawdown Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="chart-container">
          <h2 className="text-lg font-semibold text-white mb-2">{t("equityCurve")}</h2>
          <EquityCurve values={equityCurve} />
        </div>
        <div className="chart-container">
          <h2 className="text-lg font-semibold text-white mb-2">{t("drawdownChart")}</h2>
          <DrawdownChart values={drawdownSeries} />
        </div>
      </div>

      {/* Risk Checks Table */}
      <div className="chart-container">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">{t("riskCheckResults")}</h2>
          <ExportButton data={exportData} filename="atlas-risk-checks" />
        </div>
        <table className="data-table">
          <thead>
            <tr><th>{t("check")}</th><th>{t("status")}</th><th>{t("currentValue")}</th><th>{t("limit")}</th><th>{t("headroom")}</th></tr>
          </thead>
          <tbody>
            {riskChecks.map((r) => {
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
        <h2 className="text-lg font-semibold text-white mb-4">{t("positionConcentration")}</h2>
        <div className="space-y-4">
          {positions.map((pos) => (
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
          <h3 className="text-sm text-atlas-muted">{t("volatilityRegime")}</h3>
          <p className="text-2xl font-bold text-atlas-green mt-1">NORMAL</p>
          <p className="text-sm text-atlas-muted mt-2">Annualised Vol: 18.3%</p>
          <p className="text-sm text-atlas-muted">Duration: 5 days</p>
        </div>
        <div className="metric-card">
          <h3 className="text-sm text-atlas-muted">{t("riskAlerts")}</h3>
          <div className="flex items-center gap-2 mt-3">
            <div className="w-3 h-3 rounded-full bg-atlas-green" />
            <p className="text-sm text-atlas-green">{t("noAlerts")}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
