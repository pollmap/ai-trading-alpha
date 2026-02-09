"""API authentication middleware.

If ATLAS_API_KEY env var is set, requires Bearer token auth.
If not set, API is public (demo mode).
"""
import os
from http.server import BaseHTTPRequestHandler


def check_auth(handler: BaseHTTPRequestHandler) -> bool:
    """Returns True if authorized, False if rejected (sends 401)."""
    api_key = os.environ.get("ATLAS_API_KEY")
    if not api_key:
        return True  # demo mode - no auth required

    auth_header = handler.headers.get("Authorization", "")
    if auth_header == f"Bearer {api_key}":
        return True

    handler.send_response(401)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("WWW-Authenticate", "Bearer")
    handler.end_headers()
    import json
    handler.wfile.write(json.dumps({"error": "Unauthorized", "message": "Valid API key required. Set Authorization: Bearer <key>"}).encode())
    return False
