"""Risk metrics endpoint for Vercel serverless."""

from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime, timezone


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response = {
            "portfolio_var_95": 0.023,
            "max_drawdown": 0.082,
            "current_drawdown": 0.031,
            "daily_loss": 0.004,
            "volatility_regime": "normal",
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.wfile.write(json.dumps(response).encode())
