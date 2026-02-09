"""Regime detection endpoint for Vercel serverless."""

from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime, timezone

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


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
                "regimes": {
                    "KRX": {"regime": "sideways", "description": "SMA50 near SMA200, low volatility"},
                    "US": {"regime": "bull", "description": "SMA50 > SMA200, positive momentum"},
                    "CRYPTO": {"regime": "high_vol", "description": "ATR elevated, mixed signals"},
                },
                "timeline": [
                    {"week": 1, "regime": "bull"},
                    {"week": 2, "regime": "bull"},
                    {"week": 3, "regime": "sideways"},
                    {"week": 4, "regime": "sideways"},
                    {"week": 5, "regime": "bear"},
                    {"week": 6, "regime": "bear"},
                    {"week": 7, "regime": "bear"},
                    {"week": 8, "regime": "sideways"},
                    {"week": 9, "regime": "bull"},
                    {"week": 10, "regime": "bull"},
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
