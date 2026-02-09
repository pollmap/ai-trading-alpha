"""Portfolio data endpoint for Vercel serverless.

Returns mock portfolio performance data. In production, this would
query PostgreSQL/TimescaleDB for real benchmark results.
"""

from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime, timezone

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}

MOCK_PORTFOLIOS = [
    {
        "model": "deepseek",
        "architecture": "single",
        "market": "CRYPTO",
        "total_value": 108_200,
        "initial_capital": 100_000,
        "cash": 35_200,
        "return_pct": 8.2,
        "sharpe_ratio": 1.42,
        "max_drawdown": -5.1,
        "total_trades": 47,
        "win_rate": 0.58,
        "api_cost": 12.50,
    },
    {
        "model": "deepseek",
        "architecture": "multi",
        "market": "CRYPTO",
        "total_value": 111_300,
        "initial_capital": 100_000,
        "cash": 28_100,
        "return_pct": 11.3,
        "sharpe_ratio": 1.87,
        "max_drawdown": -4.2,
        "total_trades": 38,
        "win_rate": 0.62,
        "api_cost": 45.20,
    },
    {
        "model": "gemini",
        "architecture": "single",
        "market": "CRYPTO",
        "total_value": 106_800,
        "initial_capital": 100_000,
        "cash": 42_300,
        "return_pct": 6.8,
        "sharpe_ratio": 1.15,
        "max_drawdown": -6.3,
        "total_trades": 52,
        "win_rate": 0.55,
        "api_cost": 8.30,
    },
    {
        "model": "gemini",
        "architecture": "multi",
        "market": "CRYPTO",
        "total_value": 109_500,
        "initial_capital": 100_000,
        "cash": 31_200,
        "return_pct": 9.5,
        "sharpe_ratio": 1.55,
        "max_drawdown": -5.8,
        "total_trades": 41,
        "win_rate": 0.57,
        "api_cost": 32.10,
    },
    {
        "model": "claude",
        "architecture": "single",
        "market": "CRYPTO",
        "total_value": 109_100,
        "initial_capital": 100_000,
        "cash": 38_500,
        "return_pct": 9.1,
        "sharpe_ratio": 1.65,
        "max_drawdown": -4.5,
        "total_trades": 44,
        "win_rate": 0.62,
        "api_cost": 15.80,
    },
    {
        "model": "claude",
        "architecture": "multi",
        "market": "CRYPTO",
        "total_value": 112_700,
        "initial_capital": 100_000,
        "cash": 22_800,
        "return_pct": 12.7,
        "sharpe_ratio": 2.01,
        "max_drawdown": -3.8,
        "total_trades": 35,
        "win_rate": 0.66,
        "api_cost": 58.40,
    },
    {
        "model": "gpt-4o-mini",
        "architecture": "single",
        "market": "CRYPTO",
        "total_value": 105_900,
        "initial_capital": 100_000,
        "cash": 48_200,
        "return_pct": 5.9,
        "sharpe_ratio": 0.98,
        "max_drawdown": -7.1,
        "total_trades": 56,
        "win_rate": 0.52,
        "api_cost": 6.20,
    },
    {
        "model": "gpt-4o-mini",
        "architecture": "multi",
        "market": "CRYPTO",
        "total_value": 108_100,
        "initial_capital": 100_000,
        "cash": 33_400,
        "return_pct": 8.1,
        "sharpe_ratio": 1.38,
        "max_drawdown": -6.0,
        "total_trades": 43,
        "win_rate": 0.56,
        "api_cost": 22.80,
    },
    {
        "model": "buy_and_hold",
        "architecture": "baseline",
        "market": "CRYPTO",
        "total_value": 104_200,
        "initial_capital": 100_000,
        "cash": 0,
        "return_pct": 4.2,
        "sharpe_ratio": 0.72,
        "max_drawdown": -8.5,
        "total_trades": 1,
        "win_rate": 1.0,
        "api_cost": 0,
    },
]


class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self) -> None:
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()

            response = {
                "portfolios": MOCK_PORTFOLIOS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market": "CRYPTO",
            }
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
