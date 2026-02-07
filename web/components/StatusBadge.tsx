interface StatusBadgeProps {
  status: "pass" | "fail" | "warn" | "info";
  label: string;
}

const BADGE_STYLES = {
  pass: "bg-green-900/30 text-green-400",
  fail: "bg-red-900/30 text-red-400",
  warn: "bg-yellow-900/30 text-yellow-400",
  info: "bg-blue-900/30 text-blue-400",
};

export default function StatusBadge({ status, label }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${BADGE_STYLES[status]}`}
    >
      {label}
    </span>
  );
}
