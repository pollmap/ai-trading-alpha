"""Health check endpoint for Vercel serverless."""
from http.server import BaseHTTPRequestHandler
import json
import os
from datetime import datetime, timezone

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}

class handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        try:
            from api.lib.auth import check_auth
            if not check_auth(self):
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            response = {
                "status": "ok",
                "service": "ATLAS API",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "2.1.0",
                "auth_enabled": bool(os.environ.get("ATLAS_API_KEY")),
                "db_mode": "postgresql" if os.environ.get("DATABASE_URL") else "mock",
            }
            self.wfile.write(json.dumps(response).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
