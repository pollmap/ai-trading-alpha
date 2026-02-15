---
name: frontend-design
description: "Create distinctive, production-grade frontend interfaces with bold aesthetics. Avoids generic 'AI slop' design. Use when designing visually memorable pages, unique layouts, or brand-defining interfaces."
---

# Frontend Design (Anthropic Official)

Create distinctive, production-grade interfaces that avoid generic "AI slop" aesthetics. Every design should be intentional, memorable, and opinionated.

## Use this skill when

- Creating landing pages, marketing sites, or hero sections
- Designing login/auth pages that need to stand out
- Building dashboard interfaces with unique visual identity
- Choosing typography, color schemes, or animation strategies
- Making design decisions about spatial composition and layout

## Design Thinking Process

### Before writing any code:
1. **Understand the purpose** — Who sees this? What should they feel?
2. **Pick a tonal extreme** — Brutalist? Maximalist? Retro-futuristic? Minimal luxury?
3. **Decide what makes it UNFORGETTABLE** — One bold design move beats ten safe ones

## Critical Aesthetic Focus Areas

### Typography
- Choose beautiful, unique, interesting fonts
- **AVOID**: Inter, Roboto, Arial, system-ui defaults
- **PREFER**: Plus Jakarta Sans, Space Grotesk, Sora, Cabinet Grotesk, Clash Display
- Mix a display font (headings) with a clean body font
- Use font-weight variation as a design tool (200-900)
- Letter-spacing adjustments for headings (tracking-tight or tracking-wide)

### Color & Theme
- Commit to a cohesive palette — don't scatter random colors
- One dominant color, one sharp accent, neutrals for the rest
- **AVOID**: Cliche purple-to-blue gradients
- **PREFER**: Unexpected combinations — dark teal + coral, deep navy + gold, black + lime
- Dark mode isn't just inverting — redesign contrast and emphasis

### Motion & Animation
- High-impact moments > scattered micro-interactions
- Staggered reveals on page load (100ms delay per element)
- Scroll-triggered animations for below-fold content
- Smooth transitions between states (200-300ms ease-out)
- **ALWAYS** respect `prefers-reduced-motion`

### Spatial Composition
- Break the grid intentionally — asymmetry creates energy
- Overlap elements for depth (negative margins, absolute positioning)
- Generous whitespace in unexpected places
- Full-bleed sections alternating with contained content
- Diagonal lines or rotated elements for dynamism

### Details That Elevate
- Subtle noise/grain textures on backgrounds
- Layered transparencies and backdrop-blur
- Custom cursor effects on key sections
- Decorative elements (dots, lines, shapes) as accents
- Gradient borders using `border-image` or pseudo-elements

## What Makes Design "AI Slop" (AVOID)

- Default Inter/Roboto with no personality
- Purple-to-blue gradients on every section
- Identical card layouts in a 3-column grid
- Stock illustration style (isometric, flat characters)
- Predictable hero: big text left, image right
- Rounded corners on everything at the same radius
- No typographic hierarchy beyond size changes

## Implementation Checklist

- [ ] Can I describe this design in one distinctive sentence?
- [ ] Would a designer recognize this as intentional, not generated?
- [ ] Is there at least one "wow" moment on the page?
- [ ] Does the typography have personality beyond size/weight?
- [ ] Are colors cohesive and unexpected?
- [ ] Does the layout break predictable patterns?
- [ ] Are animations purposeful, not decorative noise?

## References

- Source: [anthropics/skills/frontend-design](https://github.com/anthropics/skills/tree/main/skills/frontend-design)
