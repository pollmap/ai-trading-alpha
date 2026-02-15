---
name: postgres-best-practices
description: "PostgreSQL optimization — query performance, indexing, connection pooling, schema design, and TimescaleDB hypertables. Use when writing SQL, designing schemas, or optimizing database queries."
---

# PostgreSQL Best Practices

Comprehensive PostgreSQL optimization skill covering query performance, schema design, and TimescaleDB integration.

## Use this skill when

- Writing or optimizing SQL queries
- Designing database schemas or adding tables
- Implementing indexes (B-tree, partial, composite)
- Configuring connection pooling (asyncpg, pgbouncer)
- Working with TimescaleDB hypertables and continuous aggregates
- Implementing Row-Level Security (RLS) for multi-tenancy
- Diagnosing slow queries with EXPLAIN ANALYZE

## Core Patterns (by priority)

### 1. Query Performance (Critical)

```sql
-- BAD: Missing index on frequently queried column
SELECT * FROM trades WHERE portfolio_id = 'abc';

-- GOOD: Add targeted index
CREATE INDEX CONCURRENTLY idx_trades_portfolio ON trades(portfolio_id);

-- GOOD: Partial index for common filters
CREATE INDEX idx_active_sims ON simulations(tenant_id)
  WHERE status IN ('pending', 'running');
```

### 2. Connection Management (Critical)

```python
# AsyncPG pool sizing: pool_size = num_cores * 2 + 1
engine = create_async_engine(url, pool_size=10, max_overflow=20, pool_timeout=30)
```

### 3. Multi-Tenant Isolation (Critical)

```sql
-- Row-Level Security for tenant isolation
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON portfolios
  USING (tenant_id = current_setting('app.tenant_id')::text);

-- Set tenant context per request
SET LOCAL app.tenant_id = 'tenant_abc';
```

### 4. Schema Design (High)

```sql
-- Use appropriate column types
tenant_id TEXT NOT NULL,          -- NOT uuid if you need readability
timestamp TIMESTAMPTZ NOT NULL,   -- ALWAYS with timezone
metadata JSONB DEFAULT '{}',      -- JSONB not JSON (supports indexing)
cash DOUBLE PRECISION NOT NULL,   -- NOT float4 for financial data

-- Composite index for common query patterns
CREATE INDEX idx_portfolios_tenant_market
  ON portfolios(tenant_id, market);
```

### 5. TimescaleDB Patterns

```sql
-- Convert to hypertable
SELECT create_hypertable('trades', 'timestamp', if_not_exists => TRUE);

-- Continuous aggregates for dashboards
CREATE MATERIALIZED VIEW daily_portfolio_perf
WITH (timescaledb.continuous) AS
SELECT
  time_bucket('1 day', timestamp) AS day,
  portfolio_id,
  last(total_value, timestamp) AS end_value,
  first(total_value, timestamp) AS start_value
FROM portfolio_snapshots
GROUP BY day, portfolio_id;

-- Retention policy (auto-delete old data)
SELECT add_retention_policy('trades', INTERVAL '365 days');
```

### 6. Query Optimization

```sql
-- Use EXPLAIN ANALYZE to diagnose
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM signals WHERE market = 'CRYPTO' AND timestamp > now() - interval '7 days';

-- Prefer specific columns over SELECT *
SELECT signal_id, action, confidence FROM signals WHERE ...;

-- Use EXISTS instead of COUNT for existence checks
SELECT EXISTS (SELECT 1 FROM tenants WHERE email = $1);
```

## Key Principles

1. **Index first**: Add indexes for WHERE, JOIN, ORDER BY columns
2. **CONCURRENTLY**: Always create indexes concurrently in production
3. **TIMESTAMPTZ**: Never use TIMESTAMP WITHOUT TIME ZONE
4. **Parameterized queries**: Never concatenate user input into SQL
5. **Connection pooling**: Use asyncpg with appropriate pool sizes
6. **Monitor**: Use pg_stat_statements for query analysis

## References

- Source: [antigravity-awesome-skills/postgres-best-practices](https://github.com/sickn33/antigravity-awesome-skills/tree/main/skills/postgres-best-practices)
