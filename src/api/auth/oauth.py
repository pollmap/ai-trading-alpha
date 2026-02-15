"""OAuth 2.0 flow — Google & GitHub login with HMAC-signed state."""

from __future__ import annotations

import hashlib
import secrets
from urllib.parse import urlencode

import httpx
from itsdangerous import URLSafeTimedSerializer

from config.settings import get_settings
from src.core.logging import get_logger

from .providers import PROVIDERS, OAuthProviderConfig

log = get_logger(__name__)


def _get_signer() -> URLSafeTimedSerializer:
    """HMAC signer for OAuth state tokens."""
    secret = get_settings().atlas_jwt_secret.get_secret_value()
    return URLSafeTimedSerializer(secret)


def generate_state(provider: str) -> str:
    """Create a signed, tamper-proof state token for CSRF protection."""
    signer = _get_signer()
    nonce = secrets.token_urlsafe(16)
    return signer.dumps({"provider": provider, "nonce": nonce})  # type: ignore[return-value]


def verify_state(state: str, max_age: int = 600) -> dict[str, str] | None:
    """Verify and decode the state token. Returns None if invalid/expired."""
    signer = _get_signer()
    try:
        return signer.loads(state, max_age=max_age)  # type: ignore[return-value]
    except Exception:
        log.warning("oauth_state_invalid")
        return None


def build_authorize_url(provider: str) -> str:
    """Build the OAuth authorization redirect URL."""
    settings = get_settings()
    cfg = PROVIDERS[provider]

    if provider == "google":
        client_id = settings.google_client_id
    else:
        client_id = settings.github_client_id

    state = generate_state(provider)
    callback_url = f"{settings.frontend_url}/api/auth/callback/{provider}"

    params = {
        "client_id": client_id,
        "redirect_uri": callback_url,
        "state": state,
        "scope": " ".join(cfg.scopes),
    }
    if provider == "google":
        params["response_type"] = "code"
        params["access_type"] = "offline"

    return f"{cfg.authorize_url}?{urlencode(params)}"


async def exchange_code(provider: str, code: str) -> dict[str, str]:
    """Exchange authorization code for access token + user info."""
    settings = get_settings()
    cfg = PROVIDERS[provider]

    if provider == "google":
        client_id = settings.google_client_id
        client_secret = settings.google_client_secret.get_secret_value()
    else:
        client_id = settings.github_client_id
        client_secret = settings.github_client_secret.get_secret_value()

    callback_url = f"{settings.frontend_url}/api/auth/callback/{provider}"

    # Step 1: Exchange code for token
    async with httpx.AsyncClient(timeout=15) as client:
        token_resp = await client.post(
            cfg.token_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": callback_url,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()

    access_token: str = token_data["access_token"]

    # Step 2: Fetch user info
    async with httpx.AsyncClient(timeout=15) as client:
        user_resp = await client.get(
            cfg.userinfo_url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )
        user_resp.raise_for_status()
        user_info = user_resp.json()

    # Normalize user info across providers
    return _normalize_user_info(provider, user_info)


def _normalize_user_info(
    provider: str, info: dict[str, object]
) -> dict[str, str]:
    """Normalize provider-specific user info to a common format."""
    if provider == "google":
        return {
            "provider": "google",
            "provider_id": str(info.get("sub", "")),
            "email": str(info.get("email", "")),
            "name": str(info.get("name", "")),
            "avatar_url": str(info.get("picture", "")),
        }

    # GitHub
    email = str(info.get("email", ""))
    if not email:
        # GitHub may not return email in the main response
        email = f'{info.get("login", "unknown")}@github.noreply.com'

    return {
        "provider": "github",
        "provider_id": str(info.get("id", "")),
        "email": email,
        "name": str(info.get("name") or info.get("login", "")),
        "avatar_url": str(info.get("avatar_url", "")),
    }


def generate_tenant_id(provider: str, provider_id: str) -> str:
    """Deterministic tenant_id from OAuth provider + ID."""
    raw = f"{provider}:{provider_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
