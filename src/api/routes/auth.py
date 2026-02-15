"""Authentication routes — OAuth login/callback, user info, logout."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncEngine

from config.settings import get_settings
from src.api.auth.oauth import (
    build_authorize_url,
    exchange_code,
    generate_tenant_id,
    verify_state,
)
from src.api.auth.providers import PROVIDERS
from src.api.db.tenants import TenantRepository
from src.api.deps import get_db_engine, require_auth
from src.api.middleware import create_jwt
from src.api.models.schemas import UserResponse
from src.core.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login/{provider}")
async def oauth_login(provider: str) -> RedirectResponse:
    """Redirect user to OAuth provider's authorization page."""
    if provider not in PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {provider}",
        )
    url = build_authorize_url(provider)
    return RedirectResponse(url=url, status_code=status.HTTP_302_FOUND)


@router.get("/callback/{provider}")
async def oauth_callback(
    provider: str,
    code: str,
    state: str,
    engine: AsyncEngine = Depends(get_db_engine),
) -> RedirectResponse:
    """Handle OAuth callback — exchange code, create/update user, set JWT cookie."""
    if provider not in PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown provider: {provider}",
        )

    # Verify state token (CSRF protection)
    state_data = verify_state(state)
    if state_data is None or state_data.get("provider") != provider:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state",
        )

    # Exchange code for user info
    try:
        user_info = await exchange_code(provider, code)
    except Exception as exc:
        log.error("oauth_exchange_failed", provider=provider, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="OAuth provider error",
        ) from exc

    # Upsert user in DB
    tenant_id = generate_tenant_id(provider, user_info["provider_id"])
    repo = TenantRepository(engine)
    await repo.upsert_oauth_user(
        tenant_id=tenant_id,
        name=user_info["name"],
        email=user_info["email"],
        provider=provider,
        provider_id=user_info["provider_id"],
        avatar_url=user_info.get("avatar_url", ""),
    )

    # Verify tenant is not deactivated before issuing JWT
    tenant_record = await repo.find_by_id(tenant_id)
    if tenant_record is not None and not tenant_record.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been deactivated",
        )

    # Create JWT and set cookie
    token = create_jwt(
        tenant_id,
        extra={
            "email": user_info["email"],
            "name": user_info["name"],
        },
    )

    settings = get_settings()
    response = RedirectResponse(
        url=settings.frontend_url,
        status_code=status.HTTP_302_FOUND,
    )
    response.set_cookie(
        key="atlas_token",
        value=token,
        httponly=True,
        secure=settings.atlas_env == "prod",
        samesite="lax",
        max_age=settings.atlas_jwt_expiry_hours * 3600,
        path="/",
    )

    log.info("oauth_login_success", tenant_id=tenant_id, provider=provider)
    return response


@router.get("/me", response_model=UserResponse)
async def get_me(
    tenant_id: str = Depends(require_auth),
    engine: AsyncEngine = Depends(get_db_engine),
) -> UserResponse:
    """Return the current authenticated user's info."""
    repo = TenantRepository(engine)
    tenant = await repo.find_by_id(tenant_id)
    if tenant is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserResponse(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        email=tenant.email,
        avatar_url=tenant.metadata.get("avatar_url", ""),
        plan=tenant.plan.value,
        provider=tenant.metadata.get("provider", ""),
    )


@router.post("/logout")
async def logout() -> Response:
    """Clear the JWT cookie."""
    response = Response(status_code=status.HTTP_200_OK)
    response.delete_cookie(key="atlas_token", path="/")
    return response
