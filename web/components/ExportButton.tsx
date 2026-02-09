"use client";

import { downloadCSV, downloadJSON } from "@/lib/export";
import { useI18n } from "@/lib/i18n";

interface ExportButtonProps {
  data: Record<string, unknown>[];
  filename: string;
}

export default function ExportButton({ data, filename }: ExportButtonProps) {
  const { t } = useI18n();

  return (
    <div className="flex gap-2">
      <button
        onClick={() => downloadCSV(data, filename)}
        className="px-3 py-1.5 text-xs bg-atlas-border hover:bg-atlas-border/70 text-white rounded-lg transition-colors"
      >
        {t("exportCSV")}
      </button>
      <button
        onClick={() => downloadJSON(data, filename)}
        className="px-3 py-1.5 text-xs bg-atlas-border hover:bg-atlas-border/70 text-white rounded-lg transition-colors"
      >
        {t("exportJSON")}
      </button>
    </div>
  );
}
