"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { useI18n } from "@/lib/i18n";
import { useAuth } from "@/lib/auth";
import LanguageToggle from "@/components/LanguageToggle";

type TranslationKey = "overview" | "modelComparison" | "systemStatus" | "riskDashboard" | "regimeAnalysis" | "replayViewer" | "simulation" | "customStrategy" | "optimizer" | "reports";

interface NavSection {
  title?: string;
  items: { href: string; labelKey: TranslationKey; icon: string }[];
}

const NAV_SECTIONS: NavSection[] = [
  {
    items: [
      { href: "/", labelKey: "overview", icon: "📊" },
      { href: "/model-comparison", labelKey: "modelComparison", icon: "🤖" },
      { href: "/risk", labelKey: "riskDashboard", icon: "🛡️" },
      { href: "/regime", labelKey: "regimeAnalysis", icon: "📈" },
    ],
  },
  {
    title: "v4",
    items: [
      { href: "/simulation", labelKey: "simulation", icon: "🎮" },
      { href: "/strategy", labelKey: "customStrategy", icon: "🧠" },
      { href: "/optimizer", labelKey: "optimizer", icon: "⚖️" },
      { href: "/reports", labelKey: "reports", icon: "📑" },
    ],
  },
  {
    items: [
      { href: "/system-status", labelKey: "systemStatus", icon: "⚙️" },
      { href: "/replay", labelKey: "replayViewer", icon: "⏪" },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const { t } = useI18n();
  const { user, logout } = useAuth();

  const navContent = (
    <>
      <div className="p-6 border-b border-atlas-border">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white tracking-tight">ATLAS</h1>
          <LanguageToggle />
        </div>
        <p className="text-xs text-atlas-muted mt-1">AI Trading Lab for Agent Strategy</p>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {NAV_SECTIONS.map((section, si) => (
          <div key={si}>
            {section.title && (
              <div className="px-3 pt-3 pb-1">
                <span className="text-[10px] font-bold uppercase tracking-widest text-gray-500">
                  {section.title}
                </span>
              </div>
            )}
            {section.items.map((item) => {
              const isActive = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  onClick={() => setMobileOpen(false)}
                  className={`nav-link ${isActive ? "active" : "text-atlas-muted"}`}
                >
                  <span className="text-lg">{item.icon}</span>
                  <span>{t(item.labelKey)}</span>
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      <div className="p-4 border-t border-atlas-border">
        {user ? (
          <div className="flex items-center gap-3">
            {user.avatar_url ? (
              <img
                src={user.avatar_url}
                alt={user.name}
                className="w-8 h-8 rounded-full border border-atlas-border"
              />
            ) : (
              <div className="w-8 h-8 rounded-full bg-atlas-accent flex items-center justify-center text-white text-sm font-bold">
                {user.name.charAt(0).toUpperCase()}
              </div>
            )}
            <div className="flex-1 min-w-0">
              <p className="text-sm text-white truncate">{user.name}</p>
              <p className="text-xs text-atlas-muted truncate">{user.email}</p>
            </div>
            <button
              onClick={() => logout()}
              className="p-1.5 rounded-lg text-atlas-muted hover:text-white hover:bg-atlas-border transition-colors"
              aria-label="Logout"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
              </svg>
            </button>
          </div>
        ) : (
          <>
            <p className="text-xs text-atlas-muted">4 LLMs x 2 Architectures + Buy&Hold</p>
            <p className="text-xs text-atlas-muted mt-1">10 Markets | Global Coverage</p>
          </>
        )}
        <div className="mt-3 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-atlas-green animate-pulse" />
          <span className="text-xs text-atlas-green">{t("systemOnline")}</span>
        </div>
      </div>
    </>
  );

  return (
    <>
      <button
        onClick={() => setMobileOpen(!mobileOpen)}
        className="lg:hidden fixed top-4 left-4 z-[60] p-2 rounded-lg bg-atlas-card border border-atlas-border text-white"
        aria-label="Toggle menu"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          {mobileOpen ? (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          )}
        </svg>
      </button>

      {mobileOpen && (
        <div className="lg:hidden fixed inset-0 bg-black/50 z-40" onClick={() => setMobileOpen(false)} />
      )}

      <aside
        className={`fixed left-0 top-0 h-screen w-64 bg-atlas-card border-r border-atlas-border flex flex-col z-50 transition-transform duration-300 ${
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0"
        }`}
      >
        {navContent}
      </aside>
    </>
  );
}
