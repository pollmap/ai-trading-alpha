# Analytics Module (Performance Analytics Interface) Rules

## Role

Calculate performance metrics from portfolio history and perform cross-model comparison analysis.

## Rules

- Sharpe Ratio risk-free rate: KRX -> BoK base rate, US -> 3-month T-bill, Crypto -> 0
- MDD calculation: peak-to-trough maximum decline (both % and absolute amount)
- Statistical comparison: minimum 14 days of data before drawing conclusions
- Normality test (Shapiro-Wilk) before choosing parametric/non-parametric tests
- All metrics calculated on daily return basis (trading days)
- Decimal places: returns 4 digits, ratios 2 digits, amounts 0 digits
- Cost-Adjusted Alpha (CAA) must always be included alongside traditional metrics
- Behavioral profiling uses TF-IDF from scikit-learn for reasoning keyword analysis

## Key Metrics

### Traditional
- Cumulative Return, Annualized Return, Sharpe, Sortino, MDD, Calmar, Win Rate, Profit Factor

### AI-Specific
- Signal Accuracy (directional correctness)
- Confidence Calibration (confidence vs actual correlation)
- Cost per Signal (API cost / signal count)
- Cost-Adjusted Alpha (excess return per dollar of API cost)

### Behavioral
- Action Distribution (BUY/SELL/HOLD ratios)
- Conviction Spread (confidence score distribution)
- Contrarian Score (opposing majority ratio)
- Regime Sensitivity (behavior change across market regimes)

## Statistical Methods

- Pairwise: Welch's t-test (normal) or Mann-Whitney U (non-normal)
- Multi-model: ANOVA or Kruskal-Wallis + Dunn's post-hoc
- Confidence intervals: Bootstrap resampling (1,000 iterations)
- Stability: Rolling 3-day window analysis
- Regime: Auto-classify up/down/sideways, decompose per-regime
