"""System-wide constants. All magic numbers and strings live here."""

from __future__ import annotations

# ── Market Codes ─────────────────────────────────────────────────
MARKET_KRX = "KRX"
MARKET_US = "US"
MARKET_CRYPTO = "CRYPTO"

# ── Currency Codes ───────────────────────────────────────────────
CURRENCY_KRW = "KRW"
CURRENCY_USD = "USD"
CURRENCY_USDT = "USDT"

# ── Model Provider IDs ───────────────────────────────────────────
MODEL_DEEPSEEK = "deepseek"
MODEL_GEMINI = "gemini"
MODEL_CLAUDE = "claude"
MODEL_GPT = "gpt"

# ── Agent Architecture IDs ───────────────────────────────────────
ARCH_SINGLE = "single"
ARCH_MULTI = "multi"

# ── Trading Actions ──────────────────────────────────────────────
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"

# ── Default Trading Parameters ───────────────────────────────────
DEFAULT_SLIPPAGE = 0.001            # 0.1%
DEFAULT_MAX_POSITION_WEIGHT = 0.30  # 30% of portfolio
DEFAULT_MIN_CASH_RATIO = 0.20       # 20% cash reserve
DEFAULT_TEMPERATURE = 0.0           # Deterministic for fairness

# ── Timeouts (seconds) ──────────────────────────────────────────
SINGLE_AGENT_TIMEOUT = 30
MULTI_AGENT_TIMEOUT = 120
MULTI_NODE_TIMEOUT = 60
LLM_MAX_RETRIES = 3

# ── Data Pipeline ────────────────────────────────────────────────
FORWARD_FILL_MAX_HOURS = 3          # Max stale data age before exclusion
CACHE_TTL_KRX = 60                  # seconds
CACHE_TTL_US = 60
CACHE_TTL_CRYPTO = 10

# ── Event Trigger Defaults ───────────────────────────────────────
EVENT_PRICE_CHANGE_THRESHOLD = 0.03   # 3%
EVENT_VIX_CHANGE_THRESHOLD = 0.20     # 20%
EVENT_FX_CHANGE_THRESHOLD = 0.01      # 1%
EVENT_FEAR_GREED_LOW = 20
EVENT_FEAR_GREED_HIGH = 80
EVENT_COOLDOWN_MINUTES = 15

# ── Multi-Agent ──────────────────────────────────────────────────
DEFAULT_DEBATE_ROUNDS = 2
RISK_VETO_MAX_RETRIES = 2

# ── Analytics ────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 252
BOOTSTRAP_N_RESAMPLES = 1000
ROLLING_WINDOW_DAYS = 3

# ── Dashboard Colors ─────────────────────────────────────────────
COLOR_DEEPSEEK = "#4CAF50"
COLOR_GEMINI = "#4285F4"
COLOR_CLAUDE = "#D97706"
COLOR_GPT = "#FF6B6B"
COLOR_BUY_HOLD = "#9E9E9E"

MODEL_COLORS: dict[str, str] = {
    MODEL_DEEPSEEK: COLOR_DEEPSEEK,
    MODEL_GEMINI: COLOR_GEMINI,
    MODEL_CLAUDE: COLOR_CLAUDE,
    MODEL_GPT: COLOR_GPT,
    "buy_hold": COLOR_BUY_HOLD,
}
