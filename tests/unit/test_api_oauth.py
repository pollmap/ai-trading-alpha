"""Tests for OAuth flow — state generation/verification, tenant ID generation."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.api.auth.oauth import (
    generate_state,
    generate_tenant_id,
    verify_state,
    _normalize_user_info,
)
from src.api.auth.providers import GITHUB, GOOGLE, PROVIDERS


@pytest.fixture(autouse=True)
def _mock_settings() -> None:
    mock_settings = MagicMock()
    mock_settings.atlas_jwt_secret.get_secret_value.return_value = "test-oauth-secret"
    mock_settings.atlas_jwt_expiry_hours = 24
    mock_settings.google_client_id = "google-test-id"
    mock_settings.google_client_secret.get_secret_value.return_value = "google-test-secret"
    mock_settings.github_client_id = "github-test-id"
    mock_settings.github_client_secret.get_secret_value.return_value = "github-test-secret"
    mock_settings.frontend_url = "http://localhost:3000"
    with patch("src.api.auth.oauth.get_settings", return_value=mock_settings):
        yield


class TestOAuthState:
    def test_generate_and_verify(self) -> None:
        state = generate_state("google")
        result = verify_state(state)
        assert result is not None
        assert result["provider"] == "google"
        assert "nonce" in result

    def test_verify_expired(self) -> None:
        state = generate_state("github")
        # max_age=0 should immediately expire
        result = verify_state(state, max_age=0)
        # This may or may not fail depending on timing, so we test with -1
        result = verify_state(state, max_age=-1)
        assert result is None

    def test_verify_tampered(self) -> None:
        state = generate_state("google")
        tampered = state + "x"
        result = verify_state(tampered)
        assert result is None

    def test_verify_garbage(self) -> None:
        result = verify_state("not-a-valid-state")
        assert result is None


class TestTenantIdGeneration:
    def test_deterministic(self) -> None:
        id1 = generate_tenant_id("google", "12345")
        id2 = generate_tenant_id("google", "12345")
        assert id1 == id2

    def test_different_providers(self) -> None:
        id_google = generate_tenant_id("google", "12345")
        id_github = generate_tenant_id("github", "12345")
        assert id_google != id_github

    def test_length(self) -> None:
        tid = generate_tenant_id("google", "999")
        assert len(tid) == 24

    def test_hex_chars(self) -> None:
        tid = generate_tenant_id("github", "abc")
        assert all(c in "0123456789abcdef" for c in tid)


class TestNormalizeUserInfo:
    def test_google(self) -> None:
        info = {
            "sub": "g-123",
            "email": "alice@gmail.com",
            "name": "Alice",
            "picture": "https://img.google.com/alice.jpg",
        }
        result = _normalize_user_info("google", info)
        assert result["provider"] == "google"
        assert result["provider_id"] == "g-123"
        assert result["email"] == "alice@gmail.com"
        assert result["name"] == "Alice"
        assert result["avatar_url"] == "https://img.google.com/alice.jpg"

    def test_github(self) -> None:
        info = {
            "id": 42,
            "login": "bob",
            "name": "Bob",
            "email": "bob@example.com",
            "avatar_url": "https://github.com/bob.png",
        }
        result = _normalize_user_info("github", info)
        assert result["provider"] == "github"
        assert result["provider_id"] == "42"
        assert result["email"] == "bob@example.com"

    def test_github_no_email(self) -> None:
        info = {
            "id": 99,
            "login": "nomail",
            "name": "No Mail",
        }
        result = _normalize_user_info("github", info)
        assert "nomail@github.noreply.com" == result["email"]

    def test_github_no_name_uses_login(self) -> None:
        info = {
            "id": 77,
            "login": "justlogin",
            "name": None,
            "email": "j@test.com",
        }
        result = _normalize_user_info("github", info)
        assert result["name"] == "justlogin"


class TestProviders:
    def test_google_config(self) -> None:
        assert GOOGLE.name == "google"
        assert "accounts.google.com" in GOOGLE.authorize_url
        assert "openid" in GOOGLE.scopes

    def test_github_config(self) -> None:
        assert GITHUB.name == "github"
        assert "github.com" in GITHUB.authorize_url
        assert "read:user" in GITHUB.scopes

    def test_providers_dict(self) -> None:
        assert "google" in PROVIDERS
        assert "github" in PROVIDERS
        assert len(PROVIDERS) == 2
