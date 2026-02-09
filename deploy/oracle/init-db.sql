-- ATLAS Database Schema — PostgreSQL + TimescaleDB
-- Applied automatically on first container start

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Tenants ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id       TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT UNIQUE NOT NULL,
    plan            TEXT NOT NULL DEFAULT 'free',
    api_key_hash    TEXT NOT NULL,
    is_active       BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login      TIMESTAMPTZ,
    metadata        JSONB DEFAULT '{}'
);

-- ── Portfolios ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id    TEXT PRIMARY KEY,
    tenant_id       TEXT REFERENCES tenants(tenant_id),
    model           TEXT NOT NULL,
    architecture    TEXT NOT NULL,
    market          TEXT NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    initial_capital DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_portfolios_tenant ON portfolios(tenant_id);
CREATE INDEX IF NOT EXISTS idx_portfolios_market ON portfolios(market);

-- ── Positions ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id              SERIAL PRIMARY KEY,
    portfolio_id    TEXT REFERENCES portfolios(portfolio_id),
    symbol          TEXT NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    avg_entry_price DOUBLE PRECISION NOT NULL,
    current_price   DOUBLE PRECISION NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(portfolio_id, symbol)
);

-- ── Trades (TimescaleDB hypertable) ────────────────────────────
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT NOT NULL,
    signal_id       TEXT NOT NULL,
    portfolio_id    TEXT REFERENCES portfolios(portfolio_id),
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    action          TEXT NOT NULL,
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    commission      DOUBLE PRECISION NOT NULL DEFAULT 0,
    slippage        DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl    DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (trade_id, timestamp)
);

SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);

-- ── Trading Signals (TimescaleDB hypertable) ───────────────────
CREATE TABLE IF NOT EXISTS signals (
    signal_id       TEXT NOT NULL,
    snapshot_id     TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    symbol          TEXT NOT NULL,
    market          TEXT NOT NULL,
    action          TEXT NOT NULL,
    weight          DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    reasoning       TEXT NOT NULL,
    model           TEXT NOT NULL,
    architecture    TEXT NOT NULL,
    latency_ms      DOUBLE PRECISION DEFAULT 0,
    token_usage     JSONB DEFAULT '{}',
    PRIMARY KEY (signal_id, timestamp)
);

SELECT create_hypertable('signals', 'timestamp', if_not_exists => TRUE);

-- ── LLM Call Logs (TimescaleDB hypertable) ─────────────────────
CREATE TABLE IF NOT EXISTS llm_calls (
    call_id         TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    model           TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    raw_response    TEXT NOT NULL,
    parsed_action   TEXT,
    latency_ms      DOUBLE PRECISION NOT NULL,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    cost_usd        DOUBLE PRECISION DEFAULT 0,
    tenant_id       TEXT REFERENCES tenants(tenant_id),
    PRIMARY KEY (call_id, timestamp)
);

SELECT create_hypertable('llm_calls', 'timestamp', if_not_exists => TRUE);

-- ── Portfolio Snapshots (TimescaleDB hypertable) ───────────────
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    snapshot_id     TEXT NOT NULL,
    portfolio_id    TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    total_value     DOUBLE PRECISION NOT NULL,
    cash            DOUBLE PRECISION NOT NULL,
    positions_json  JSONB DEFAULT '{}',
    PRIMARY KEY (snapshot_id, timestamp)
);

SELECT create_hypertable('portfolio_snapshots', 'timestamp', if_not_exists => TRUE);

-- ── Simulations ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS simulations (
    simulation_id   TEXT PRIMARY KEY,
    tenant_id       TEXT REFERENCES tenants(tenant_id),
    name            TEXT NOT NULL,
    config_json     JSONB NOT NULL,
    status          TEXT NOT NULL DEFAULT 'created',
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    total_cycles    INTEGER DEFAULT 0,
    error_message   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_simulations_tenant ON simulations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_simulations_status ON simulations(status);

-- ── Custom Strategies ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id     TEXT PRIMARY KEY,
    tenant_id       TEXT REFERENCES tenants(tenant_id),
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    prompt_template TEXT NOT NULL,
    model           TEXT NOT NULL DEFAULT 'claude',
    markets         JSONB DEFAULT '["US"]',
    risk_params     JSONB DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'draft',
    version         INTEGER DEFAULT 1,
    tags            JSONB DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_strategies_tenant ON strategies(tenant_id);

-- ── Usage Metering ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS usage_events (
    id              SERIAL,
    tenant_id       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    quantity        INTEGER DEFAULT 1,
    metadata        JSONB DEFAULT '{}',
    PRIMARY KEY (id, timestamp)
);

SELECT create_hypertable('usage_events', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_usage_tenant ON usage_events(tenant_id, timestamp DESC);

-- ── Market Data Cache ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id     TEXT NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    market          TEXT NOT NULL,
    symbols_json    JSONB NOT NULL,
    macro_json      JSONB DEFAULT '{}',
    news_json       JSONB DEFAULT '[]',
    PRIMARY KEY (snapshot_id, timestamp)
);

SELECT create_hypertable('market_snapshots', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_market ON market_snapshots(market, timestamp DESC);

-- ── Continuous Aggregates for Analytics ─────────────────────────

-- Daily portfolio performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_portfolio_performance
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    portfolio_id,
    last(total_value, timestamp) AS end_value,
    first(total_value, timestamp) AS start_value,
    max(total_value) AS high_value,
    min(total_value) AS low_value
FROM portfolio_snapshots
GROUP BY day, portfolio_id
WITH NO DATA;

-- Daily trade summary
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_trade_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    portfolio_id,
    count(*) AS trade_count,
    sum(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) AS buy_count,
    sum(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) AS sell_count,
    sum(realized_pnl) AS total_realized_pnl,
    sum(commission) AS total_commission
FROM trades
GROUP BY day, portfolio_id
WITH NO DATA;

-- Grants
GRANT ALL ON ALL TABLES IN SCHEMA public TO atlas;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO atlas;
