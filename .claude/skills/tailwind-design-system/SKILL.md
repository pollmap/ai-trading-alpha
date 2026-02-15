---
name: tailwind-design-system
description: "Build scalable design systems with Tailwind CSS — design tokens, component libraries, dark mode, responsive patterns. Use when creating or extending Tailwind-based UI components."
---

# Tailwind Design System

Production-ready design systems using Tailwind CSS with design tokens, theming, and component standardization.

## Use this skill when

- Creating component libraries with Tailwind
- Implementing design tokens and theming
- Building responsive and accessible components
- Standardizing UI patterns across pages
- Setting up dark mode and color schemes
- Migrating to or extending Tailwind CSS

## Design Token System

### tailwind.config.ts
```typescript
const config: Config = {
  darkMode: 'class',
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        atlas: {
          bg: '#0F1117',
          card: '#1A1D2E',
          border: '#2A2D3E',
          muted: '#94A3B8',
          green: '#10B981',
          red: '#EF4444',
          blue: '#3B82F6',
          amber: '#F59E0B',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Plus Jakarta Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
      },
    },
  },
};
```

## Component Patterns

### Card Component
```tsx
function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-atlas-card border border-atlas-border rounded-xl p-6 ${className ?? ''}`}>
      {children}
    </div>
  );
}
```

### MetricCard
```tsx
function MetricCard({ label, value, trend, icon }: MetricCardProps) {
  const trendColor = trend > 0 ? 'text-atlas-green' : trend < 0 ? 'text-atlas-red' : 'text-atlas-muted';
  return (
    <Card>
      <div className="flex items-center justify-between">
        <span className="text-atlas-muted text-sm">{label}</span>
        <span className="text-lg">{icon}</span>
      </div>
      <div className="mt-2 text-2xl font-bold text-white">{value}</div>
      <div className={`mt-1 text-sm ${trendColor}`}>
        {trend > 0 ? '+' : ''}{trend}%
      </div>
    </Card>
  );
}
```

### Button Variants
```tsx
const variants = {
  primary: 'bg-atlas-blue hover:bg-blue-600 text-white',
  danger: 'bg-atlas-red hover:bg-red-600 text-white',
  ghost: 'bg-transparent hover:bg-atlas-border text-atlas-muted hover:text-white',
  outline: 'border border-atlas-border hover:border-atlas-blue text-atlas-muted hover:text-white',
} as const;
```

## Responsive Patterns

```tsx
// Mobile-first responsive
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
  {metrics.map(m => <MetricCard key={m.label} {...m} />)}
</div>

// Sidebar: hidden on mobile, fixed on desktop
<aside className="fixed left-0 top-0 h-screen w-64 -translate-x-full lg:translate-x-0 transition-transform">
```

## Dark Mode

```tsx
// Toggle in tailwind.config: darkMode: 'class'
// Apply on <html> element: <html className="dark">

// Component-level dark styles
<div className="bg-white dark:bg-atlas-card text-gray-900 dark:text-white">
```

## Spacing Scale

| Name | Value | Use |
|------|-------|-----|
| 1 | 4px | Icon gaps |
| 2 | 8px | Tight padding |
| 3 | 12px | Button padding |
| 4 | 16px | Card inner padding |
| 6 | 24px | Section padding |
| 8 | 32px | Large gaps |
| 12 | 48px | Page margins |

## Key Principles

1. **Utility-first**: Compose utilities, extract components only when repeated 3+ times
2. **Design tokens**: All colors, fonts, spacing defined in config — never hardcode
3. **Responsive**: Mobile-first, use `md:` and `lg:` breakpoints
4. **Consistency**: Same spacing, radius, shadow across all components
5. **Accessibility**: Focus rings, contrast ratios, keyboard navigation

## References

- Source: [antigravity-awesome-skills/tailwind-design-system](https://github.com/sickn33/antigravity-awesome-skills/tree/main/skills/tailwind-design-system)
