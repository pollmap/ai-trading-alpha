"""Health check endpoint for Vercel serverless."""

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
            "status": "ok",
            "service": "ATLAS API",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
        }
        self.wfile.write(json.dumps(response).encode())
