"use client";

import MetricCard from "@/components/MetricCard";
import ExportButton from "@/components/ExportButton";
import { useI18n } from "@/lib/i18n";

const API_STATUS = [
  { provider: "DeepSeek R1", status: "Online", latency: "1.2s", calls: 1247, cost: "$12.50", errors: 3 },
  { provider: "Gemini 2.5 Pro", status: "Online", latency: "0.8s", calls: 1189, cost: "$8.30", errors: 1 },
  { provider: "Claude Sonnet 4.5", status: "Online", latency: "1.5s", calls: 1312, cost: "$15.80", errors: 5 },
  { provider: "GPT-4o-mini", status: "Online", latency: "0.6s", calls: 1456, cost: "$6.20", errors: 2 },
];

const SERVICES = [
  { name: "PostgreSQL + TimescaleDB", status: "Running", uptime: "99.9%", detail: "32 GB used" },
  { name: "Redis Cache", status: "Running", uptime: "100%", detail: "2.1 GB / 8 GB" },
  { name: "Data Scheduler", status: "Running", uptime: "99.7%", detail: "Next: 5m 23s" },
  { name: "Benchmark Runner", status: "Idle", uptime: "N/A", detail: "Last: 2h ago" },
];

export default function SystemStatusPage() {
  const { t } = useI18n();

  const providerExportData = API_STATUS.map((api) => ({
    provider: api.provider,
    status: api.status,
    avg_latency: api.latency,
    api_calls: api.calls,
    cost: api.cost,
    errors: api.errors,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">{t("sysStatusTitle")}</h1>
        <p className="text-atlas-muted mt-1">{t("sysStatusDesc")}</p>
      </div>

      {/* Top Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard label={t("totalApiCalls")} value="5,204" delta={t("last24h")} />
        <MetricCard label={t("totalApiCost")} value="$42.80" delta={t("last24h")} />
        <MetricCard label={t("avgLatency")} value="1.02s" delta="-0.1s" deltaType="positive" />
        <MetricCard label={t("errorRate")} value="0.21%" delta="11 / 5,204" deltaType="negative" />
      </div>

      {/* LLM Provider Status */}
      <div className="chart-container">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">{t("llmProviderStatus")}</h2>
          <ExportButton data={providerExportData} filename="atlas-llm-provider-status" />
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>{t("provider")}</th>
              <th>{t("status")}</th>
              <th>{t("latency")}</th>
              <th>{t("apiCalls")}</th>
              <th>{t("cost")}</th>
              <th>{t("errors")}</th>
            </tr>
          </thead>
          <tbody>
            {API_STATUS.map((api) => (
              <tr key={api.provider} className="hover:bg-atlas-border/30">
                <td className="font-medium text-white">{api.provider}</td>
                <td>
                  <span className="badge-pass">{api.status}</span>
                </td>
                <td>{api.latency}</td>
                <td>{api.calls.toLocaleString()}</td>
                <td>{api.cost}</td>
                <td className={api.errors > 3 ? "text-atlas-red" : "text-atlas-muted"}>
                  {api.errors}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Infrastructure */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">{t("infrastructure")}</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {SERVICES.map((svc) => (
            <div key={svc.name} className="bg-atlas-bg rounded-lg p-4 border border-atlas-border flex items-center justify-between">
              <div>
                <h3 className="text-sm font-medium text-white">{svc.name}</h3>
                <p className="text-xs text-atlas-muted mt-1">{svc.detail}</p>
              </div>
              <div className="text-right">
                <span className={`text-xs px-2 py-1 rounded-full ${
                  svc.status === "Running" ? "bg-green-900/30 text-green-400" :
                  svc.status === "Idle" ? "bg-yellow-900/30 text-yellow-400" :
                  "bg-red-900/30 text-red-400"
                }`}>
                  {svc.status}
                </span>
                {svc.uptime !== "N/A" && (
                  <p className="text-xs text-atlas-muted mt-1">{svc.uptime} uptime</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Errors */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">{t("recentErrors")}</h2>
        <div className="space-y-2">
          {[
            { time: "14:32 UTC", provider: "Claude", error: "Rate limit exceeded", resolution: "Auto-retry after 60s" },
            { time: "11:15 UTC", provider: "DeepSeek", error: "Timeout (30s)", resolution: "Retried successfully" },
            { time: "08:45 UTC", provider: "GPT-4o-mini", error: "Invalid JSON response", resolution: "Fallback to HOLD" },
          ].map((err, i) => (
            <div key={i} className="bg-atlas-bg rounded-lg p-3 border border-atlas-border flex items-center gap-4">
              <span className="text-xs text-atlas-muted whitespace-nowrap">{err.time}</span>
              <span className="badge-warn">{err.provider}</span>
              <span className="text-sm text-white flex-1">{err.error}</span>
              <span className="text-xs text-atlas-green">{err.resolution}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
