"use client";

import { useI18n } from "@/lib/i18n";

export default function LanguageToggle() {
  const { locale, setLocale } = useI18n();

  return (
    <button
      onClick={() => setLocale(locale === "ko" ? "en" : "ko")}
      className="px-2 py-1 text-xs rounded border border-atlas-border text-atlas-muted hover:text-white hover:border-atlas-accent transition-colors"
    >
      {locale === "ko" ? "EN" : "한국어"}
    </button>
  );
}
