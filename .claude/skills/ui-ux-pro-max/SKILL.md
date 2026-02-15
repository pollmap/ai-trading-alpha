---
name: ui-ux-pro-max
description: "Professional UI/UX design intelligence — 50+ styles, 97 color palettes, 57 font pairings, SaaS/fintech industry rules. Auto-activates when requesting UI/UX design, login pages, dashboards, or component styling."
---

# UI/UX Pro Max

Comprehensive design intelligence system for creating production-grade, accessible, and visually distinctive interfaces.

## Use this skill when

- Designing login/signup pages
- Building dashboard layouts and data visualization UIs
- Creating component libraries or design systems
- Choosing color palettes, typography, or visual styles
- Implementing dark mode themes
- Ensuring accessibility compliance (WCAG 2.1 AA)

## Design System Generation

When designing any UI, always start with a design system:

### 1. Analyze Requirements
- Product type (SaaS dashboard, fintech, trading platform)
- Target audience (traders, analysts, general users)
- Technology stack (Next.js + Tailwind + React)
- Brand personality (professional, bold, minimal, futuristic)

### 2. Color Palette (SaaS/Fintech)
```
Primary:    #3B82F6 (Blue-500) — trust, reliability
Secondary:  #10B981 (Emerald-500) — profit, growth
Danger:     #EF4444 (Red-500) — loss, alerts
Warning:    #F59E0B (Amber-500) — caution
Background: #0F1117 (Dark surface)
Card:       #1A1D2E (Elevated surface)
Border:     #2A2D3E (Subtle divider)
Text:       #F8FAFC (Primary text)
Muted:      #94A3B8 (Secondary text)
```

### 3. Typography
```
Headings:   Inter (600/700 weight) or Plus Jakarta Sans
Body:       Inter (400/500 weight)
Monospace:  JetBrains Mono (for code, data, numbers)
Scale:      text-xs(12) / text-sm(14) / text-base(16) / text-lg(18) / text-xl(20) / text-2xl(24) / text-3xl(30)
```

### 4. Component Patterns

**Login Page:**
- Centered card on dark gradient background
- OAuth buttons with provider icons (SVG, not emoji)
- Clear visual hierarchy: logo → title → subtitle → buttons
- Loading states on all interactive elements

**Dashboard Cards:**
- Consistent padding (p-6), rounded corners (rounded-xl)
- Subtle border (border-atlas-border) on dark background
- MetricCard: icon + label + value + trend indicator
- Hover state: slight elevation or border highlight

**Data Tables:**
- Sticky header, zebra striping optional
- Sortable columns with arrow indicators
- Pagination or virtual scrolling for large datasets
- Row hover highlight

**Charts:**
- Consistent color coding (green=profit, red=loss)
- Tooltip on hover with formatted values
- Responsive container sizing
- Legend placement: top or right

## Critical UX Rules

### Accessibility (CRITICAL)
- Minimum 4.5:1 contrast ratio for text
- All interactive elements keyboard-accessible
- Focus indicators visible and clear
- `aria-label` on icon-only buttons
- `role` attributes on custom components
- Respect `prefers-reduced-motion`

### Interaction (HIGH)
- `cursor-pointer` on ALL clickable elements
- Loading spinners/skeletons for async operations
- Disabled states visually distinct (opacity-50 + cursor-not-allowed)
- Toast notifications for actions (success/error)
- Confirm dialogs for destructive actions

### Layout (HIGH)
- Mobile-first responsive design
- Sidebar: fixed on desktop, overlay on mobile
- Content area: max-width for readability on wide screens
- Consistent spacing scale (4/8/12/16/24/32/48)

### Performance (MEDIUM)
- Lazy load below-fold content
- Optimize images (WebP, proper sizing)
- Skeleton screens over spinners where possible
- Debounce search inputs (300ms)

## Anti-Patterns to Avoid

- No emoji as icons — use Heroicons, Lucide, or custom SVG
- No generic sans-serif (Arial, Helvetica) — use Inter or Plus Jakarta Sans
- No inconsistent spacing or padding
- No unformatted numbers (use Intl.NumberFormat)
- No layouts that break below 375px width
- No forms without validation feedback
- No color-only status indicators (add text/icon)

## References

- Source: [nextlevelbuilder/ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill)
