"use client";

import { useState } from "react";
import { useI18n } from "@/lib/i18n";

const MOCK_REPORT_HISTORY = [
  { id: "rpt-001", type: "excel", date: "2026-02-09", size: "2.4 MB", portfolios: 27, status: "ready" },
  { id: "rpt-002", type: "word", date: "2026-02-08", size: "1.8 MB", portfolios: 27, status: "ready" },
  { id: "rpt-003", type: "pdf", date: "2026-02-07", size: "890 KB", portfolios: 9, status: "ready" },
];

const formatIcons: Record<string, string> = {
  excel: "📊",
  word: "📝",
  pdf: "📄",
};

export default function ReportsPage() {
  const { t } = useI18n();
  const [generating, setGenerating] = useState<string | null>(null);

  const handleGenerate = (format: string) => {
    setGenerating(format);
    setTimeout(() => setGenerating(null), 2000);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">{t("reportTitle")}</h1>
        <p className="text-gray-400 mt-1">{t("reportDesc")}</p>
      </div>

      {/* Generate Buttons */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { format: "excel", label: t("downloadExcel"), desc: "Multi-sheet workbook with charts, equity curves, trade log", color: "from-green-600 to-green-700", ext: ".xlsx" },
          { format: "word", label: t("downloadWord"), desc: "Professional report with executive summary, per-model analysis", color: "from-blue-600 to-blue-700", ext: ".docx" },
          { format: "pdf", label: t("downloadPDF"), desc: "Compact summary with key metrics and highlights", color: "from-red-600 to-red-700", ext: ".pdf" },
        ].map(item => (
          <button key={item.format} onClick={() => handleGenerate(item.format)}
            disabled={generating !== null}
            className={`bg-gradient-to-r ${item.color} rounded-xl p-6 text-left hover:opacity-90 transition-all disabled:opacity-50`}>
            <div className="text-3xl mb-2">{formatIcons[item.format]}</div>
            <div className="font-semibold text-lg">{item.label}</div>
            <div className="text-sm text-white/70 mt-1">{item.desc}</div>
            <div className="text-xs text-white/50 mt-2">{item.ext}</div>
            {generating === item.format && (
              <div className="mt-3 flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white/50 border-t-white rounded-full animate-spin" />
                <span className="text-sm">Generating...</span>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Report Configuration */}
      <div className="bg-atlas-panel border border-atlas-border rounded-xl p-5">
        <h3 className="font-semibold mb-4">{t("generateReport")}</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">{t("markets")}</label>
            <select className="w-full bg-atlas-dark border border-atlas-border rounded px-2 py-1.5 text-sm">
              <option>All Markets</option>
              <option>US Only</option>
              <option>Asia (JPX, SSE, HKEX)</option>
              <option>Europe (EURONEXT, LSE)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Period</label>
            <select className="w-full bg-atlas-dark border border-atlas-border rounded px-2 py-1.5 text-sm">
              <option>Last 30 days</option>
              <option>Last 7 days</option>
              <option>Last 90 days</option>
              <option>All time</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Include</label>
            <div className="flex flex-col gap-1">
              <label className="flex items-center gap-1 text-xs">
                <input type="checkbox" defaultChecked className="accent-blue-500" /> Trade Log
              </label>
              <label className="flex items-center gap-1 text-xs">
                <input type="checkbox" defaultChecked className="accent-blue-500" /> Equity Curves
              </label>
            </div>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Format</label>
            <div className="flex flex-col gap-1">
              <label className="flex items-center gap-1 text-xs">
                <input type="checkbox" defaultChecked className="accent-blue-500" /> Charts
              </label>
              <label className="flex items-center gap-1 text-xs">
                <input type="checkbox" defaultChecked className="accent-blue-500" /> Recommendations
              </label>
            </div>
          </div>
        </div>
      </div>

      {/* Report History */}
      <div className="bg-atlas-panel border border-atlas-border rounded-xl p-5">
        <h3 className="font-semibold mb-4">Report History</h3>
        <div className="space-y-2">
          {MOCK_REPORT_HISTORY.map(rpt => (
            <div key={rpt.id} className="flex items-center justify-between bg-atlas-dark rounded-lg px-4 py-3">
              <div className="flex items-center gap-3">
                <span className="text-xl">{formatIcons[rpt.type]}</span>
                <div>
                  <div className="text-sm font-medium">
                    ATLAS Benchmark Report — {rpt.type.toUpperCase()}
                  </div>
                  <div className="text-xs text-gray-400">
                    {rpt.date} · {rpt.portfolios} portfolios · {rpt.size}
                  </div>
                </div>
              </div>
              <button className="px-3 py-1 bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 rounded text-xs transition-colors">
                Download
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
