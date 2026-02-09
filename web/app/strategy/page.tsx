"use client";

import { useState } from "react";
import { useI18n } from "@/lib/i18n";
import ExportButton from "@/components/ExportButton";

const MOCK_STRATEGIES = [
  { id: "strat-001", name: "Momentum Scanner", model: "claude", markets: ["US", "CRYPTO"], status: "active", version: 3, trades: 48, returnPct: 12.5 },
  { id: "strat-002", name: "Value Contrarian", model: "deepseek", markets: ["JPX", "HKEX"], status: "draft", version: 1, trades: 0, returnPct: 0 },
  { id: "strat-003", name: "News Sentiment Alpha", model: "gpt", markets: ["US", "EURONEXT", "LSE"], status: "active", version: 5, trades: 82, returnPct: 8.3 },
  { id: "strat-004", name: "Bond-Equity Rotator", model: "gemini", markets: ["BOND", "US", "COMMODITIES"], status: "archived", version: 2, trades: 35, returnPct: -1.2 },
];

const TEMPLATE_VARS = ["{market_data}", "{portfolio_state}", "{regime}", "{news}", "{macro}", "{custom_indicators}"];

const statusStyles: Record<string, string> = {
  active: "bg-green-500/20 text-green-400",
  draft: "bg-gray-500/20 text-gray-400",
  archived: "bg-orange-500/20 text-orange-400",
};

export default function StrategyPage() {
  const { t } = useI18n();
  const [showBuilder, setShowBuilder] = useState(false);
  const [template, setTemplate] = useState(
    "Analyze the current market data:\n{market_data}\n\nPortfolio state:\n{portfolio_state}\n\nMarket regime: {regime}\n\nBased on this analysis, determine the optimal trading action."
  );

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-bold">{t("stratTitle")}</h1>
          <p className="text-gray-400 mt-1">{t("stratDesc")}</p>
        </div>
        <div className="flex gap-2">
          <ExportButton data={MOCK_STRATEGIES} filename="strategies" />
          <button onClick={() => setShowBuilder(!showBuilder)}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors">
            + {t("createStrategy")}
          </button>
        </div>
      </div>

      {showBuilder && (
        <div className="bg-atlas-panel border border-atlas-border rounded-xl p-6 space-y-4">
          <h3 className="text-lg font-semibold">{t("createStrategy")}</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">{t("stratName")}</label>
              <input type="text" className="w-full bg-atlas-dark border border-atlas-border rounded-lg px-3 py-2 text-sm" />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">{t("selectModel")}</label>
              <select className="w-full bg-atlas-dark border border-atlas-border rounded-lg px-3 py-2 text-sm">
                <option value="claude">Claude</option>
                <option value="gpt">GPT-4o-mini</option>
                <option value="deepseek">DeepSeek</option>
                <option value="gemini">Gemini</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">{t("promptTemplate")}</label>
            <div className="flex flex-wrap gap-1 mb-2">
              {TEMPLATE_VARS.map(v => (
                <button key={v} onClick={() => setTemplate(prev => prev + "\n" + v)}
                  className="text-xs bg-blue-600/20 text-blue-400 px-2 py-0.5 rounded hover:bg-blue-600/30 transition-colors">
                  {v}
                </button>
              ))}
            </div>
            <textarea value={template} onChange={e => setTemplate(e.target.value)}
              className="w-full bg-atlas-dark border border-atlas-border rounded-lg px-3 py-2 text-sm font-mono h-40 resize-y" />
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-2">{t("riskParams")}</label>
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="block text-xs text-gray-500">{t("maxPositionWeight")}</label>
                <input type="number" defaultValue={0.30} step={0.05} min={0} max={1}
                  className="w-full bg-atlas-dark border border-atlas-border rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-500">{t("stopLoss")}</label>
                <input type="number" defaultValue={5} step={1} min={1} max={50}
                  className="w-full bg-atlas-dark border border-atlas-border rounded px-2 py-1 text-sm" />
              </div>
              <div>
                <label className="block text-xs text-gray-500">{t("takeProfit")}</label>
                <input type="number" defaultValue={15} step={1} min={1} max={100}
                  className="w-full bg-atlas-dark border border-atlas-border rounded px-2 py-1 text-sm" />
              </div>
            </div>
          </div>

          <div>
            <label className="block text-sm text-gray-400 mb-1">{t("selectMarkets")}</label>
            <div className="flex flex-wrap gap-2">
              {["US", "KRX", "CRYPTO", "JPX", "SSE", "HKEX", "EURONEXT", "LSE", "BOND", "COMMODITIES"].map(m => (
                <label key={m} className="flex items-center gap-1 text-sm bg-atlas-dark border border-atlas-border rounded px-2 py-1 cursor-pointer hover:border-blue-500 transition-colors">
                  <input type="checkbox" className="accent-blue-500" defaultChecked={m === "US"} />
                  {m}
                </label>
              ))}
            </div>
          </div>

          <div className="flex gap-2">
            <button className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg text-sm font-medium transition-colors">{t("validate")}</button>
            <button className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-sm font-medium transition-colors">{t("createStrategy")}</button>
          </div>
        </div>
      )}

      <div className="grid gap-4">
        {MOCK_STRATEGIES.map(strat => (
          <div key={strat.id} className="bg-atlas-panel border border-atlas-border rounded-xl p-5">
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-3">
                  <h3 className="font-semibold">{strat.name}</h3>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${statusStyles[strat.status]}`}>
                    {t(strat.status as "draft" | "active" | "archived")}
                  </span>
                  <span className="text-xs text-gray-500">v{strat.version}</span>
                </div>
                <p className="text-sm text-gray-400 mt-1">
                  {strat.model.toUpperCase()} · {strat.markets.join(", ")} · {strat.trades} {t("trades")}
                </p>
              </div>
              <div className="text-right">
                <div className={`text-lg font-bold ${strat.returnPct >= 0 ? "text-green-400" : "text-red-400"}`}>
                  {strat.returnPct >= 0 ? "+" : ""}{strat.returnPct}%
                </div>
                <div className="text-xs text-gray-500">{t("returnPct")}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
