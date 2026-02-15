"""JWT authentication middleware for FastAPI."""

from __future__ import annotations

from fastapi import Cookie, HTTPException, Request, status

from config.settings import get_settings
from src.core.logging import get_logger
from src.saas.tenant import JWTManager

log = get_logger(__name__)

_jwt_manager: JWTManager | None = None


def _get_jwt() -> JWTManager:
    """Lazy-init singleton JWTManager."""
    global _jwt_manager  # noqa: PLW0603
    if _jwt_manager is None:
        settings = get_settings()
        _jwt_manager = JWTManager(
            secret=settings.atlas_jwt_secret.get_secret_value(),
            expiry_hours=settings.atlas_jwt_expiry_hours,
        )
    return _jwt_manager


def create_jwt(tenant_id: str, extra: dict[str, str] | None = None) -> str:
    """Create a JWT token for the given tenant."""
    return _get_jwt().create_token(tenant_id, extra_claims=extra)


def verify_jwt(token: str) -> dict[str, str] | None:
    """Verify a JWT token and return the payload."""
    return _get_jwt().verify_token(token)


async def get_current_user(
    request: Request,
    atlas_token: str | None = Cookie(default=None),
) -> dict[str, str]:
    """Extract and verify the JWT from the cookie or Authorization header.

    Returns the JWT payload dict with at least 'sub' (tenant_id).
    """
    token: str | None = atlas_token

    # Fallback: check Authorization header (for API clients)
    if not token:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    payload = verify_jwt(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return payload
