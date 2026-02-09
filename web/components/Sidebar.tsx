"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: "📊" },
  { href: "/model-comparison", label: "Model Comparison", icon: "🤖" },
  { href: "/system-status", label: "System Status", icon: "⚙️" },
  { href: "/risk", label: "Risk Dashboard", icon: "🛡️" },
  { href: "/regime", label: "Regime Analysis", icon: "📈" },
  { href: "/replay", label: "Replay Viewer", icon: "⏪" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const navContent = (
    <>
      <div className="p-6 border-b border-atlas-border">
        <h1 className="text-2xl font-bold text-white tracking-tight">ATLAS</h1>
        <p className="text-xs text-atlas-muted mt-1">
          AI Trading Lab for Agent Strategy
        </p>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={() => setMobileOpen(false)}
              className={`nav-link ${isActive ? "active" : "text-atlas-muted"}`}
            >
              <span className="text-lg">{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-atlas-border">
        <p className="text-xs text-atlas-muted">
          4 LLMs x 2 Architectures + Buy&Hold
        </p>
        <p className="text-xs text-atlas-muted mt-1">
          Markets: KRX | US | CRYPTO
        </p>
        <div className="mt-3 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-atlas-green animate-pulse" />
          <span className="text-xs text-atlas-green">System Online</span>
        </div>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile hamburger */}
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

      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar */}
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
