"""OAuth provider configuration for Google and GitHub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OAuthProviderConfig:
    """Immutable OAuth provider configuration."""

    name: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: list[str]


GOOGLE = OAuthProviderConfig(
    name="google",
    authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    userinfo_url="https://www.googleapis.com/oauth2/v3/userinfo",
    scopes=["openid", "email", "profile"],
)

GITHUB = OAuthProviderConfig(
    name="github",
    authorize_url="https://github.com/login/oauth/authorize",
    token_url="https://github.com/login/oauth/access_token",
    userinfo_url="https://api.github.com/user",
    scopes=["read:user", "user:email"],
)

PROVIDERS: dict[str, OAuthProviderConfig] = {
    "google": GOOGLE,
    "github": GITHUB,
}
