"""Data provider abstraction.

Uses MockProvider by default. If DATABASE_URL env var is set,
uses PostgreSQLProvider for real data queries.
"""
import os
from datetime import datetime, timezone


class DataProvider:
    """Abstract data provider."""

    def get_portfolios(self, market: str = "CRYPTO") -> dict:
        raise NotImplementedError

    def get_risk_metrics(self) -> dict:
        raise NotImplementedError

    def get_regime_data(self) -> dict:
        raise NotImplementedError


class MockProvider(DataProvider):
    """Returns mock data for demo/development."""

    def get_portfolios(self, market: str = "CRYPTO") -> dict:
        portfolios = [
            {"model": "deepseek", "architecture": "single", "market": market, "total_value": 108200, "initial_capital": 100000, "cash": 35200, "return_pct": 8.2, "sharpe_ratio": 1.42, "max_drawdown": -5.1, "total_trades": 47, "win_rate": 0.58, "api_cost": 12.50},
            {"model": "deepseek", "architecture": "multi", "market": market, "total_value": 111300, "initial_capital": 100000, "cash": 28100, "return_pct": 11.3, "sharpe_ratio": 1.87, "max_drawdown": -4.2, "total_trades": 38, "win_rate": 0.62, "api_cost": 45.20},
            {"model": "gemini", "architecture": "single", "market": market, "total_value": 106800, "initial_capital": 100000, "cash": 42300, "return_pct": 6.8, "sharpe_ratio": 1.15, "max_drawdown": -6.3, "total_trades": 52, "win_rate": 0.55, "api_cost": 8.30},
            {"model": "gemini", "architecture": "multi", "market": market, "total_value": 109500, "initial_capital": 100000, "cash": 31200, "return_pct": 9.5, "sharpe_ratio": 1.55, "max_drawdown": -5.8, "total_trades": 41, "win_rate": 0.57, "api_cost": 32.10},
            {"model": "claude", "architecture": "single", "market": market, "total_value": 109100, "initial_capital": 100000, "cash": 38500, "return_pct": 9.1, "sharpe_ratio": 1.65, "max_drawdown": -4.5, "total_trades": 44, "win_rate": 0.62, "api_cost": 15.80},
            {"model": "claude", "architecture": "multi", "market": market, "total_value": 112700, "initial_capital": 100000, "cash": 22800, "return_pct": 12.7, "sharpe_ratio": 2.01, "max_drawdown": -3.8, "total_trades": 35, "win_rate": 0.66, "api_cost": 58.40},
            {"model": "gpt-4o-mini", "architecture": "single", "market": market, "total_value": 105900, "initial_capital": 100000, "cash": 48200, "return_pct": 5.9, "sharpe_ratio": 0.98, "max_drawdown": -7.1, "total_trades": 56, "win_rate": 0.52, "api_cost": 6.20},
            {"model": "gpt-4o-mini", "architecture": "multi", "market": market, "total_value": 108100, "initial_capital": 100000, "cash": 33400, "return_pct": 8.1, "sharpe_ratio": 1.38, "max_drawdown": -6.0, "total_trades": 43, "win_rate": 0.56, "api_cost": 22.80},
            {"model": "buy_and_hold", "architecture": "baseline", "market": market, "total_value": 104200, "initial_capital": 100000, "cash": 0, "return_pct": 4.2, "sharpe_ratio": 0.72, "max_drawdown": -8.5, "total_trades": 1, "win_rate": 1.0, "api_cost": 0},
        ]
        return {"portfolios": portfolios, "timestamp": datetime.now(timezone.utc).isoformat(), "market": market}

    def get_risk_metrics(self) -> dict:
        return {
            "portfolio_var_95": 0.023, "max_drawdown": 0.082, "current_drawdown": 0.031,
            "daily_loss": 0.004, "volatility_regime": "normal",
            "checks": [
                {"name": "position_limit", "passed": True, "value": 0.22, "limit": 0.30},
                {"name": "cash_reserve", "passed": True, "value": 0.352, "limit": 0.20},
                {"name": "confidence", "passed": True, "value": 0.75, "limit": 0.30},
                {"name": "drawdown_circuit_breaker", "passed": True, "value": 0.031, "limit": 0.15},
                {"name": "daily_loss_limit", "passed": True, "value": 0.004, "limit": 0.05},
            ],
            "positions": [
                {"symbol": "BTCUSDT", "weight": 0.22, "value": 22000},
                {"symbol": "ETHUSDT", "weight": 0.18, "value": 18000},
                {"symbol": "AAPL", "weight": 0.15, "value": 15000},
                {"symbol": "005930", "weight": 0.10, "value": 10000},
            ],
            "equity_curve": [100000, 100800, 101200, 100500, 101800, 103200, 102100, 104500, 105200, 104800, 106100, 107300, 106800, 108200],
            "drawdown_series": [0, -0.002, -0.005, -0.012, -0.003, 0, -0.011, 0, -0.002, -0.008, 0, -0.003, -0.008, 0],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_regime_data(self) -> dict:
        return {
            "regimes": {
                "KRX": {"regime": "sideways", "description": "SMA50 near SMA200, low volatility"},
                "US": {"regime": "bull", "description": "SMA50 > SMA200, positive momentum"},
                "CRYPTO": {"regime": "high_vol", "description": "ATR elevated, mixed signals"},
            },
            "timeline": [
                {"week": 1, "regime": "bull"}, {"week": 2, "regime": "bull"},
                {"week": 3, "regime": "sideways"}, {"week": 4, "regime": "sideways"},
                {"week": 5, "regime": "bear"}, {"week": 6, "regime": "bear"},
                {"week": 7, "regime": "bear"}, {"week": 8, "regime": "sideways"},
                {"week": 9, "regime": "bull"}, {"week": 10, "regime": "bull"},
            ],
            "performance_by_regime": [
                {"model": "DeepSeek", "bull": 8.2, "sideways": 1.2, "bear": -3.1},
                {"model": "Gemini", "bull": 7.5, "sideways": 0.8, "bear": -2.5},
                {"model": "Claude", "bull": 9.1, "sideways": 2.1, "bear": -1.8},
                {"model": "GPT-4o-mini", "bull": 6.8, "sideways": -0.5, "bear": -4.2},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class PostgreSQLProvider(DataProvider):
    """Queries real PostgreSQL/TimescaleDB. Requires DATABASE_URL env var."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url

    def _get_conn(self):
        try:
            import psycopg2
            return psycopg2.connect(self._url)
        except ImportError:
            raise RuntimeError("psycopg2 not installed. Run: pip install psycopg2-binary")

    def get_portfolios(self, market: str = "CRYPTO") -> dict:
        # In production, this queries the real DB
        # For now, falls back to mock if DB is unreachable
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT model, architecture, market, total_value, initial_capital, cash, "
                "return_pct, sharpe_ratio, max_drawdown, total_trades, win_rate, api_cost "
                "FROM portfolio_snapshots WHERE market = %s ORDER BY model, architecture",
                (market,)
            )
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return MockProvider().get_portfolios(market)
            cols = ["model", "architecture", "market", "total_value", "initial_capital", "cash",
                    "return_pct", "sharpe_ratio", "max_drawdown", "total_trades", "win_rate", "api_cost"]
            portfolios = [dict(zip(cols, row)) for row in rows]
            return {"portfolios": portfolios, "timestamp": datetime.now(timezone.utc).isoformat(), "market": market}
        except Exception:
            return MockProvider().get_portfolios(market)

    def get_risk_metrics(self) -> dict:
        # Falls back to mock for now
        return MockProvider().get_risk_metrics()

    def get_regime_data(self) -> dict:
        return MockProvider().get_regime_data()


def get_provider() -> DataProvider:
    """Factory: returns PostgreSQLProvider if DATABASE_URL is set, else MockProvider."""
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return PostgreSQLProvider(db_url)
    return MockProvider()
