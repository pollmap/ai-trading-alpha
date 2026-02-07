interface MetricCardProps {
  label: string;
  value: string;
  delta?: string;
  deltaType?: "positive" | "negative" | "neutral";
}

export default function MetricCard({
  label,
  value,
  delta,
  deltaType = "neutral",
}: MetricCardProps) {
  const deltaColor =
    deltaType === "positive"
      ? "text-atlas-green"
      : deltaType === "negative"
        ? "text-atlas-red"
        : "text-atlas-muted";

  return (
    <div className="metric-card">
      <p className="text-sm text-atlas-muted">{label}</p>
      <p className="text-2xl font-bold text-white mt-1">{value}</p>
      {delta && (
        <p className={`text-sm mt-1 ${deltaColor}`}>{delta}</p>
      )}
    </div>
  );
}
