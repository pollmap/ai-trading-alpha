"use client";

import { useState } from "react";
import MetricCard from "@/components/MetricCard";
import ExportButton from "@/components/ExportButton";
import { useI18n } from "@/lib/i18n";

const MOCK_SESSIONS = [
  { id: "benchmark-2026-01-15-full", events: 1247, snapshots: 48, signals: 384, trades: 156, date: "2026-01-15" },
  { id: "benchmark-2026-01-14-full", events: 1189, snapshots: 48, signals: 372, trades: 148, date: "2026-01-14" },
  { id: "benchmark-2026-01-13-partial", events: 623, snapshots: 24, signals: 186, trades: 72, date: "2026-01-13" },
];

const MOCK_EVENTS = [
  { seq: 1, type: "session_start", time: "09:00:00", detail: "Benchmark started" },
  { seq: 2, type: "snapshot", time: "09:00:05", detail: "KRX market data collected (10 symbols)" },
  { seq: 3, type: "signal", time: "09:00:12", detail: "DeepSeek/Single → BUY 005930 (conf=0.82)" },
  { seq: 4, type: "risk_check", time: "09:00:12", detail: "All 5 checks PASSED" },
  { seq: 5, type: "trade", time: "09:00:13", detail: "BUY 10 shares 005930 @ 71,200 KRW" },
  { seq: 6, type: "signal", time: "09:00:15", detail: "Gemini/Single → HOLD (conf=0.65)" },
  { seq: 7, type: "signal", time: "09:00:18", detail: "Claude/Multi → BUY 005930 (conf=0.88)" },
  { seq: 8, type: "risk_check", time: "09:00:18", detail: "All 5 checks PASSED" },
  { seq: 9, type: "trade", time: "09:00:19", detail: "BUY 8 shares 005930 @ 71,250 KRW" },
  { seq: 10, type: "portfolio_update", time: "09:00:20", detail: "9 portfolios updated" },
];

const EVENT_COLORS: Record<string, string> = {
  session_start: "bg-blue-900/30 text-blue-400",
  session_end: "bg-blue-900/30 text-blue-400",
  snapshot: "bg-purple-900/30 text-purple-400",
  signal: "bg-indigo-900/30 text-indigo-400",
  risk_check: "bg-cyan-900/30 text-cyan-400",
  trade: "bg-green-900/30 text-green-400",
  portfolio_update: "bg-yellow-900/30 text-yellow-400",
  error: "bg-red-900/30 text-red-400",
};

export default function ReplayViewerPage() {
  const { t } = useI18n();
  const [selectedSession, setSelectedSession] = useState(MOCK_SESSIONS[0]);
  const [filterTypes, setFilterTypes] = useState<string[]>(["signal", "trade", "risk_check"]);

  const filteredEvents = MOCK_EVENTS.filter(
    (e) => filterTypes.length === 0 || filterTypes.includes(e.type)
  );

  const toggleFilter = (type: string) => {
    setFilterTypes((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type]
    );
  };

  const eventExportData = filteredEvents.map((e) => ({
    seq: e.seq,
    type: e.type,
    time: e.time,
    detail: e.detail,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">{t("replayTitle")}</h1>
        <p className="text-atlas-muted mt-1">{t("replayDesc")}</p>
      </div>

      {/* Session Selector */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-3">{t("selectSession")}</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {MOCK_SESSIONS.map((sess) => (
            <button
              key={sess.id}
              onClick={() => setSelectedSession(sess)}
              className={`text-left p-3 rounded-lg border transition-colors ${
                selectedSession.id === sess.id
                  ? "border-atlas-accent bg-atlas-accent/10"
                  : "border-atlas-border hover:border-atlas-muted"
              }`}
            >
              <p className="text-sm font-medium text-white">{sess.date}</p>
              <p className="text-xs text-atlas-muted mt-1">{sess.id}</p>
              <p className="text-xs text-atlas-muted">{sess.events} events</p>
            </button>
          ))}
        </div>
      </div>

      {/* Session Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard label={t("totalEvents")} value={selectedSession.events.toLocaleString()} />
        <MetricCard label={t("snapshots")} value={selectedSession.snapshots.toString()} />
        <MetricCard label={t("signals")} value={selectedSession.signals.toString()} />
        <MetricCard label={t("trades")} value={selectedSession.trades.toString()} />
      </div>

      {/* Event Filter */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-3">{t("eventFilter")}</h2>
        <div className="flex flex-wrap gap-2">
          {Object.keys(EVENT_COLORS).map((type) => (
            <button
              key={type}
              onClick={() => toggleFilter(type)}
              className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
                filterTypes.includes(type)
                  ? `${EVENT_COLORS[type]} border-transparent`
                  : "border-atlas-border text-atlas-muted hover:text-white"
              }`}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Event Timeline */}
      <div className="chart-container">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">{t("eventTimeline")}</h2>
          <ExportButton data={eventExportData} filename="atlas-replay-events" />
        </div>
        <div className="space-y-2">
          {filteredEvents.map((event) => (
            <div
              key={event.seq}
              className="flex items-center gap-3 p-3 rounded-lg bg-atlas-bg border border-atlas-border/50 hover:border-atlas-border transition-colors"
            >
              <span className="text-xs text-atlas-muted font-mono w-8">#{event.seq}</span>
              <span className="text-xs text-atlas-muted font-mono w-20">{event.time}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full w-32 text-center ${EVENT_COLORS[event.type] || "bg-gray-900/30 text-gray-400"}`}>
                {event.type}
              </span>
              <span className="text-sm text-white flex-1">{event.detail}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Decision Trail Example */}
      <div className="chart-container">
        <h2 className="text-lg font-semibold text-white mb-4">{t("decisionTrail")}</h2>
        <div className="space-y-3">
          {[
            { step: 1, label: "Snapshot", detail: "Market data collected (005930: 71,200 KRW)", color: "border-purple-500" },
            { step: 2, label: "Signal", detail: "DeepSeek/Single → BUY 005930 (weight=0.15, conf=0.82)", color: "border-indigo-500" },
            { step: 3, label: "Risk Check", detail: "5/5 checks PASSED (position 8% < 30%, cash 72% > 20%)", color: "border-cyan-500" },
            { step: 4, label: "Trade", detail: "BUY 10 shares @ 71,200 KRW (commission: 71 KRW)", color: "border-green-500" },
            { step: 5, label: "Portfolio", detail: "Cash 92,880 KRW → Position 005930: 10 shares", color: "border-yellow-500" },
          ].map((step) => (
            <div key={step.step} className={`border-l-4 ${step.color} pl-4 py-2`}>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-atlas-accent">Step {step.step}</span>
                <span className="text-xs text-atlas-muted">{step.label}</span>
              </div>
              <p className="text-sm text-white mt-1">{step.detail}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
