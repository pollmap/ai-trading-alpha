"""Tests for JWT middleware — create/verify tokens, extract user."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.middleware import create_jwt, verify_jwt, get_current_user


@pytest.fixture(autouse=True)
def _reset_jwt_manager() -> None:
    """Reset the global JWTManager between tests."""
    import src.api.middleware as mod
    mod._jwt_manager = None


@pytest.fixture(autouse=True)
def _mock_settings() -> None:
    """Mock get_settings for all tests in this module."""
    mock_settings = MagicMock()
    mock_settings.atlas_jwt_secret.get_secret_value.return_value = "test-secret-key"
    mock_settings.atlas_jwt_expiry_hours = 24
    with patch("src.api.middleware.get_settings", return_value=mock_settings):
        yield


class TestCreateVerifyJWT:
    def test_create_and_verify(self) -> None:
        token = create_jwt("tenant-123")
        payload = verify_jwt(token)
        assert payload is not None
        assert payload["sub"] == "tenant-123"

    def test_create_with_extra_claims(self) -> None:
        token = create_jwt("t1", extra={"email": "a@b.com", "name": "Alice"})
        payload = verify_jwt(token)
        assert payload is not None
        assert payload["email"] == "a@b.com"
        assert payload["name"] == "Alice"

    def test_invalid_token(self) -> None:
        result = verify_jwt("invalid.token.here")
        assert result is None

    def test_tampered_token(self) -> None:
        token = create_jwt("t1")
        # Tamper with the token
        parts = token.split(".")
        parts[1] = parts[1] + "x"
        tampered = ".".join(parts)
        result = verify_jwt(tampered)
        assert result is None

    def test_wrong_secret(self) -> None:
        token = create_jwt("t1")

        # Reset and use different secret
        import src.api.middleware as mod
        mod._jwt_manager = None

        mock_settings = MagicMock()
        mock_settings.atlas_jwt_secret.get_secret_value.return_value = "different-secret"
        mock_settings.atlas_jwt_expiry_hours = 24

        with patch("src.api.middleware.get_settings", return_value=mock_settings):
            result = verify_jwt(token)
            assert result is None


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_from_cookie(self) -> None:
        token = create_jwt("tenant-abc")
        request = MagicMock()
        request.headers.get.return_value = ""

        result = await get_current_user(request, atlas_token=token)
        assert result["sub"] == "tenant-abc"

    @pytest.mark.asyncio
    async def test_from_bearer_header(self) -> None:
        token = create_jwt("tenant-xyz")
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"

        result = await get_current_user(request, atlas_token=None)
        assert result["sub"] == "tenant-xyz"

    @pytest.mark.asyncio
    async def test_no_token_raises_401(self) -> None:
        from fastapi import HTTPException

        request = MagicMock()
        request.headers.get.return_value = ""

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request, atlas_token=None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_token_raises_401(self) -> None:
        from fastapi import HTTPException

        request = MagicMock()
        request.headers.get.return_value = ""

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(request, atlas_token="bad-token")
        assert exc_info.value.status_code == 401
