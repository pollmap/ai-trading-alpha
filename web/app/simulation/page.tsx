"use client";

import { useState } from "react";
import { useI18n } from "@/lib/i18n";
import ExportButton from "@/components/ExportButton";

const MOCK_SIMULATIONS = [
  { id: "sim-001", name: "US Bull Market Test", status: "completed", cycles: 720, markets: ["US", "CRYPTO"], startDate: "2026-01-01", endDate: "2026-01-31", infinite: false },
  { id: "sim-002", name: "Global Stress Test", status: "running", cycles: 245, markets: ["US", "JPX", "EURONEXT", "CRYPTO"], startDate: "2026-02-01", endDate: null, infinite: true },
  { id: "sim-003", name: "Asia Focus", status: "paused", cycles: 120, markets: ["JPX", "SSE", "HKEX"], startDate: "2026-02-05", endDate: "2026-03-05", infinite: false },
];

const statusColors: Record<string, string> = {
  running: "bg-green-500/20 text-green-400",
  paused: "bg-yellow-500/20 text-yellow-400",
  completed: "bg-blue-500/20 text-blue-400",
  stopped: "bg-red-500/20 text-red-400",
  created: "bg-gray-500/20 text-gray-400",
};

export default function SimulationPage() {
  const { t } = useI18n();
  const [showCreate, setShowCreate] = useState(false);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold">{t("simTitle")}</h1>
          <p className="text-gray-400 mt-1">{t("simDesc")}</p>
        </div>
        <div className="flex gap-2">
          <ExportButton data={MOCK_SIMULATIONS} filename="simulations" />
          <button
            onClick={() => setShowCreate(!showCreate)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
          >
            + {t("createSim")}
          </button>
        </div>
      </div>

      {showCreate && (
        <div className="bg-atlas-panel border border-atlas-border rounded-xl p-6 space-y-4">
          <h3 className="text-lg font-semibold">{t("createSim")}</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">{t("simName")}</label>
              <input type="text" className="w-full bg-atlas-dark border border-atlas-border rounded-lg px-3 py-2 text-sm" placeholder="My Simulation" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">{t("interval")}</label>
              <select className="w-full bg-atlas-dark border border-atlas-border rounded-lg px-3 py-2 text-sm">
                <option value="3600">1 hour</option>
                <option value="1800">30 min</option>
                <option value="86400">1 day</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">{t("startDate")}</label>
              <input type="date" className="w-full bg-atlas-dark border border-atlas-border rounded-lg px-3 py-2 text-sm" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">{t("endDate")}</label>
              <input type="date" className="w-full bg-atlas-dark border border-atlas-border rounded-lg px-3 py-2 text-sm" placeholder="Leave empty for infinite" />
            </div>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">{t("markets")}</label>
            <div className="flex flex-wrap gap-2">
              {["US", "KRX", "CRYPTO", "JPX", "SSE", "HKEX", "EURONEXT", "LSE", "BOND", "COMMODITIES"].map(m => (
                <label key={m} className="flex items-center gap-1 text-sm bg-atlas-dark border border-atlas-border rounded-lg px-3 py-1.5 cursor-pointer hover:border-blue-500 transition-colors">
                  <input type="checkbox" className="accent-blue-500" defaultChecked={["US", "CRYPTO"].includes(m)} />
                  {m}
                </label>
              ))}
            </div>
          </div>
          <button className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition-colors">
            {t("createSim")}
          </button>
        </div>
      )}

      <div className="space-y-3">
        {MOCK_SIMULATIONS.map(sim => (
          <div key={sim.id} className="bg-atlas-panel border border-atlas-border rounded-xl p-5">
            <div className="flex justify-between items-center">
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="font-semibold">{sim.name}</h3>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${statusColors[sim.status]}`}>
                    {t(sim.status as "running" | "paused" | "completed" | "stopped")}
                  </span>
                  {sim.infinite && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-400">
                      {t("infiniteMode")}
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-400 mt-1">
                  {sim.markets.join(", ")} · {sim.cycles} {t("cycles")} · {sim.startDate}
                  {sim.endDate ? ` → ${sim.endDate}` : " → ∞"}
                </p>
              </div>
              <div className="flex gap-2">
                {sim.status === "running" && (
                  <button className="px-3 py-1.5 bg-yellow-600/20 text-yellow-400 hover:bg-yellow-600/30 rounded-lg text-xs transition-colors">
                    {t("pause")}
                  </button>
                )}
                {sim.status === "paused" && (
                  <button className="px-3 py-1.5 bg-green-600/20 text-green-400 hover:bg-green-600/30 rounded-lg text-xs transition-colors">
                    {t("resume")}
                  </button>
                )}
                {(sim.status === "running" || sim.status === "paused") && (
                  <>
                    <button className="px-3 py-1.5 bg-red-600/20 text-red-400 hover:bg-red-600/30 rounded-lg text-xs transition-colors">
                      {t("stop")}
                    </button>
                    <button className="px-3 py-1.5 bg-purple-600/20 text-purple-400 hover:bg-purple-600/30 rounded-lg text-xs transition-colors">
                      {t("injectScenario")}
                    </button>
                  </>
                )}
              </div>
            </div>
            {sim.status === "running" && (
              <div className="mt-3">
                <div className="w-full bg-atlas-dark rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full animate-pulse" style={{ width: "60%" }} />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
