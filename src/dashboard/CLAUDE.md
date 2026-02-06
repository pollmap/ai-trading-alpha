# Dashboard Module Rules

## Role

Streamlit-based real-time dashboard. Visualize benchmark progress and performance.

## Page Structure

1. **Overview** — Overall summary (9 portfolio equity curves + core metrics cards)
2. **System Status** — Benchmark progress, agent health, error logs, next cycle countdown
3. **Model Comparison** — 4 models x 2 architectures matrix (heatmap, radar chart)
4. **Market View** — Per-market detail (KRX/US/Crypto tabs)
5. **Agent Detail** — Individual agent decision timeline + reasoning text + Agent Personality
6. **Cost Monitor** — Per-model API cost charts + Cost-Adjusted Alpha comparison

## Rules

- Charts: Plotly only (no matplotlib — interactive required)
- Color scheme:
  - DeepSeek: #4CAF50 (green)
  - Gemini: #4285F4 (blue)
  - Claude: #D97706 (amber)
  - GPT: #FF6B6B (coral)
  - Buy&Hold: #9E9E9E (gray)
  - Single agent: solid line
  - Multi agent: dashed line
- Auto-refresh: 30-second interval (st.rerun or streamlit-autorefresh)
- Heavy queries: `@st.cache_data(ttl=60)` caching mandatory
- Mobile support: not required — desktop only
- All time displays in KST (data stored as UTC, convert on display)
