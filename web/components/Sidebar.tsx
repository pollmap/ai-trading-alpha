"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

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

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 bg-atlas-card border-r border-atlas-border flex flex-col z-50">
      {/* Logo */}
      <div className="p-6 border-b border-atlas-border">
        <h1 className="text-2xl font-bold text-white tracking-tight">ATLAS</h1>
        <p className="text-xs text-atlas-muted mt-1">
          AI Trading Lab for Agent Strategy
        </p>
      </div>

      {/* Navigation */}
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
              className={`nav-link ${isActive ? "active" : "text-atlas-muted"}`}
            >
              <span className="text-lg">{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-atlas-border">
        <p className="text-xs text-atlas-muted">
          4 LLMs × 2 Architectures + Buy&Hold
        </p>
        <p className="text-xs text-atlas-muted mt-1">
          Markets: KRX | US | CRYPTO
        </p>
        <div className="mt-3 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-atlas-green animate-pulse" />
          <span className="text-xs text-atlas-green">System Online</span>
        </div>
      </div>
    </aside>
  );
}
