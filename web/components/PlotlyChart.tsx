"use client";

import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const CHART_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#94a3b8", size: 12 },
  margin: { l: 50, r: 20, t: 30, b: 40 },
  xaxis: { gridcolor: "#2d3148", zerolinecolor: "#2d3148" },
  yaxis: { gridcolor: "#2d3148", zerolinecolor: "#2d3148" },
  showlegend: false,
};

const CHART_CONFIG = {
  displayModeBar: false,
  responsive: true,
};

interface EquityCurveProps {
  values: number[];
  title?: string;
}

export function EquityCurve({ values, title = "Equity Curve" }: EquityCurveProps) {
  return (
    <div className="w-full h-[300px]">
      <Plot
        data={[
          {
            y: values,
            type: "scatter" as const,
            mode: "lines" as const,
            fill: "tozeroy",
            fillcolor: "rgba(99, 102, 241, 0.1)",
            line: { color: "#6366f1", width: 2 },
            hovertemplate: "$%{y:,.0f}<extra></extra>",
          },
        ]}
        layout={{
          ...CHART_LAYOUT,
          title: { text: title, font: { color: "#e2e8f0", size: 14 }, x: 0 },
          yaxis: { ...CHART_LAYOUT.yaxis, tickprefix: "$", tickformat: ",d" },
          xaxis: { ...CHART_LAYOUT.xaxis, title: "Day" },
        }}
        config={CHART_CONFIG}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}

interface DrawdownChartProps {
  values: number[];
  title?: string;
}

export function DrawdownChart({ values, title = "Drawdown" }: DrawdownChartProps) {
  return (
    <div className="w-full h-[250px]">
      <Plot
        data={[
          {
            y: values.map((v) => v * 100),
            type: "scatter" as const,
            mode: "lines" as const,
            fill: "tozeroy",
            fillcolor: "rgba(239, 68, 68, 0.15)",
            line: { color: "#ef4444", width: 2 },
            hovertemplate: "%{y:.2f}%<extra></extra>",
          },
        ]}
        layout={{
          ...CHART_LAYOUT,
          title: { text: title, font: { color: "#e2e8f0", size: 14 }, x: 0 },
          yaxis: { ...CHART_LAYOUT.yaxis, ticksuffix: "%", rangemode: "nonpositive" as const },
          xaxis: { ...CHART_LAYOUT.xaxis, title: "Day" },
        }}
        config={CHART_CONFIG}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}

interface RegimeTimelineChartProps {
  timeline: { week: number; regime: string }[];
  title?: string;
}

const REGIME_COLORS: Record<string, string> = {
  bull: "#22c55e",
  bear: "#ef4444",
  sideways: "#6b7280",
  high_vol: "#eab308",
  crash: "#dc2626",
};

export function RegimeTimelineChart({ timeline, title = "Regime Timeline" }: RegimeTimelineChartProps) {
  return (
    <div className="w-full h-[200px]">
      <Plot
        data={[
          {
            x: timeline.map((t) => `W${t.week}`),
            y: timeline.map(() => 1),
            type: "bar" as const,
            marker: {
              color: timeline.map((t) => REGIME_COLORS[t.regime] || "#6b7280"),
            },
            text: timeline.map((t) => t.regime.toUpperCase()),
            textposition: "inside" as const,
            textfont: { color: "white", size: 11 },
            hovertemplate: "Week %{x}: %{text}<extra></extra>",
          },
        ]}
        layout={{
          ...CHART_LAYOUT,
          title: { text: title, font: { color: "#e2e8f0", size: 14 }, x: 0 },
          yaxis: { ...CHART_LAYOUT.yaxis, visible: false },
          xaxis: { ...CHART_LAYOUT.xaxis },
          bargap: 0.05,
        }}
        config={CHART_CONFIG}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}

interface ModelBarChartProps {
  models: { name: string; value: number }[];
  title?: string;
  suffix?: string;
  color?: string;
}

export function ModelBarChart({ models, title = "", suffix = "%", color = "#6366f1" }: ModelBarChartProps) {
  return (
    <div className="w-full h-[250px]">
      <Plot
        data={[
          {
            x: models.map((m) => m.name),
            y: models.map((m) => m.value),
            type: "bar" as const,
            marker: { color, opacity: 0.8 },
            text: models.map((m) => `${m.value}${suffix}`),
            textposition: "outside" as const,
            textfont: { color: "#e2e8f0", size: 12 },
            hovertemplate: "%{x}: %{y}" + suffix + "<extra></extra>",
          },
        ]}
        layout={{
          ...CHART_LAYOUT,
          title: { text: title, font: { color: "#e2e8f0", size: 14 }, x: 0 },
          yaxis: { ...CHART_LAYOUT.yaxis, ticksuffix: suffix },
        }}
        config={CHART_CONFIG}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
