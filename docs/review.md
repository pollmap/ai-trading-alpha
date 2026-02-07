# ATLAS - Project Review, Comparison & Refinements

## One-Sentence Summary

"A benchmark platform that runs 4 LLMs in single/multi-agent configurations across Korean stocks, US stocks, and crypto markets via real-time virtual trading, then statistically compares their performance."

---

## Comparison with Existing Research

| Aspect | TradingAgents | Agent Market Arena (AMA) | StockBench | DeepFund | **ATLAS** |
|--------|--------------|--------------------------|------------|----------|-----------|
| Markets | US stocks only | BTC, ETH, TSLA, BMRN | US stocks | US stocks | **KRX + US + Crypto** |
| Evaluation | Backtesting | Real-time forward | Forward (historical window) | Real-time forward | **Real-time forward** |
| Model comparison | GPT-4o centric | 5 models | 6 models | 4 models | **4 models** |
| Architecture comparison | Multi only | Various frameworks | Single only | Multi only | **Single vs Multi cross-comparison** |
| Korean market | No | No | No | No | **Yes** |
| Open source | Yes | No | Partial | Partial | **Yes (planned)** |

---

## Differentiation (Real Strengths)

### 1. Korean Market Inclusion is an Academic Blue Ocean

Of 13 LLM trading papers from 2024-2025, zero covered the Korean market. AI-Trader was the only one testing a non-US market (Chinese A-shares). Evaluating LLM agents on KRX data is virtually unexplored territory, making this viable for research reports or academic presentations.

### 2. Architecture x Model 2-Axis Evaluation Design

Most existing research examines "which model is best" OR "is multi-agent better than single" separately. Few combine them. AMA comes closest but uses different agent frameworks per model, making variable control impossible. ATLAS uses identical prompts + identical data + identical pipeline, changing only the model — much cleaner variable control.

### 3. Multi-Agent "Debate" Structure

The Bull/Bear researcher debate pattern is validated by TradingAgents. Combining this with DeepSeek-R1's CoT reasoning (extractable via `reasoning_content`) enables uniquely transparent analysis of "why this decision was made."

### 4. Cost-Adjusted Alpha (CAA) — Novel Metric

No existing paper quantifies multi-agent excess returns per dollar of API cost. ATLAS explicitly measures:

```
Cost-Adjusted Alpha (CAA) = (agent return - B&H return) / total API cost
```

This directly answers: "Does the extra API cost of multi-agent justify the performance gain?"

---

## Honest Limitations & Mitigations

### 1. Statistical Significance Challenge

**Concern:** Real-time forward tests can't repeat the same market conditions 30 times.

**Rebuttal:** Demanding "30 independent trials" applies to backtesting paradigms. DeepFund published 1 real-time test (31 days) and was accepted at NeurIPS. The value of forward testing is measuring *uncontaminated* future prediction ability, not repeatability.

**Mitigation (what we added):**
- **Phase-based evaluation:** 72h -> 2 weeks -> 1 month+
- **Rolling window analysis:** Split total period into 3-day windows, measure per-window performance variance for stability
- **Bootstrap resampling:** 1,000 virtual periods from daily returns for CI estimation
- **Regime tagging:** Auto-classify up/down/sideways, decompose per-regime performance

### 2. Only 4 Models (GPT-4 Absent)

**Rebuttal:** This is actually a strength. Existing papers are GPT-biased. Using DeepSeek (China), Gemini (Google), Claude (Anthropic) creates a "non-OpenAI model financial reasoning" research angle that's underexplored.

**Mitigation:** Added GPT-4o-mini as reference baseline. Cheapest model ($0.15/M input), enables indirect comparison with existing GPT-4o results from literature.

### 3. News/Sentiment Data Gap

**Concern:** Valid. Trading agents without news data are "driving blind." Especially problematic for multi-agent sentiment/news analysts receiving empty data.

**Mitigation (implemented):**
- Korean: Google News RSS (free, no scraping needed)
- US: Finnhub News API (free 60 calls/min)
- Crypto: CryptoPanic API (free tier)
- LLM-as-sentiment-analyzer: Feed 10 headlines to LLM, get -1 to +1 score
- Graceful degradation: empty news list doesn't halt cycles

### 4. 30-Minute Cycle ≠ "Real-Time"

**Rebuttal:** ATLAS is not an HFT simulator. The goal is measuring "LLM ability to read market situations and make rational decisions." 30-minute intervals are actually more frequent than actual PB/asset manager rebalancing (daily~weekly). Shorter intervals (5 min) would cause LLM overfitting to noise.

**Mitigation:** Added Event-Driven Triggers for price spikes (>=3%), VIX surges (>=20%), FX moves (>=1%), extreme Fear&Greed (<20 or >80). Normal cycles at 30/15 min, emergency cycles only on anomalies. 15-min cooldown prevents API cost explosion.

### 5. Multi-Agent Cost-Effectiveness Uncertain

**Rebuttal:** This is not a weakness — it IS the research question. "Is multi-agent worth 10x the API cost?" is exactly what ATLAS is designed to answer quantitatively via the CAA metric. The 2025 paper (arXiv:2505.18286) noted multi-agent advantages diminish as frontier models improve, making this timing-sensitive research.

### 6. Simple Slippage Model

**Concern:** Fixed 0.1% slippage is unrealistic for small-caps or crypto altcoins. Real markets have nonlinear slippage based on order book depth.

**Mitigation:** V1 keeps the simple model (sufficient for large-cap focus). V2 can incorporate Binance order book depth data for dynamic slippage. This is a known simplification, not a fatal flaw.

---

## Practical Applications

### Career Portfolio
- Open-source on GitHub -> interview material ("here's my project" with live link)
- "AI trading agent benchmark" directly relevant to securities firm AI adoption trends
- Results report formatted as research note -> analyst competency demonstration

### Academic Use
- Viable as undergraduate thesis topic (LLM trading benchmark on Korean market)
- Submittable to Korean Finance Association undergraduate paper competitions
- KRX-based results have scarcity value in domestic academic community

### CUFA (Finance Club) Use
- Research project presentation (SMIC-level research orientation)
- Inter-club collaboration content (with 3F at Chungnam National)
- New member training: "How AI makes investment decisions" visual demo

### CBDC (Crypto Club) Use
- Crypto market results as standalone digital asset research
- "How AI judges BTC trades" -> seminar material

---

## Biggest Risk

**Completion risk.** 5-week roadmap, but full-stack projects via Claude Code typically take 1.5-2x estimated time. API integration surprises, LangGraph debugging, and cross-market data format inconsistencies will cause delays.

**Mitigation: Aggressive MVP scoping.**

```
MVP (Week 1~3, MUST complete):
├── Crypto market ONLY (24/7, no market hours logic needed)
├── 3 models single agent ONLY (no multi-agent)
├── Binance data ONLY (most stable API)
├── Basic dashboard (equity curve + core metrics)
└── 72h continuous execution capable

Even MVP alone delivers:
→ "3 LLM crypto trading ability comparison" research complete
→ Dashboard screenshots for portfolio material
→ Publishable open-source project on GitHub

Extension 1 (Week 4):
├── Multi-agent pipeline
├── GPT-4o-mini baseline
└── Cost-Adjusted Alpha analysis

Extension 2 (Week 5+):
├── KRX + US market addition
├── News/sentiment integration
├── Statistical comparison (bootstrap CI, etc.)
├── Event-driven triggers
└── Behavioral profiling
```

---

## Refinement Summary

| Aspect | Original | Refined |
|--------|----------|---------|
| Model count | 3 (DeepSeek, Gemini, Claude) | **4+1** (+GPT-4o-mini baseline, +Buy&Hold) |
| Benchmark period | 72 hours | **Phased: 72h -> 2 weeks -> 1 month+** |
| News data | Placeholder | **RSS + Finnhub + CryptoPanic (real implementation)** |
| Cycle method | Fixed interval only | **Fixed + Event-driven triggers** |
| Key metrics | Traditional finance only | **+Cost-Adjusted Alpha, +Behavioral Profiling** |
| Reproducibility | Snapshots only | **+Full prompt + raw response recording** |
| Statistical analysis | Basic t-test | **+Rolling window, +Bootstrap CI, +Regime decomposition** |
| MVP scope | All markets simultaneously | **Crypto single agent first -> gradual expansion** |
| DB tables | 5 | **6** (+llm_call_logs) |
| Directory additions | — | **+gpt_adapter.py, +call_logger.py, +behavioral_profiler.py, +news_adapter.py (real)** |
