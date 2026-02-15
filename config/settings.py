"""ATLAS global settings — loaded from environment variables via .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All configuration flows through this class. Never read env vars directly."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Environment ──────────────────────────────────────────────
    atlas_env: Literal["dev", "prod"] = "dev"

    # ── LLM Providers ────────────────────────────────────────────
    deepseek_api_key: SecretStr = SecretStr("")
    gemini_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")

    # ── Korean Market Data ───────────────────────────────────────
    kis_app_key: SecretStr = SecretStr("")
    kis_app_secret: SecretStr = SecretStr("")
    kis_account_no: str = ""
    bok_api_key: SecretStr = SecretStr("")
    opendart_api_key: SecretStr = SecretStr("")

    # ── US Market Data ───────────────────────────────────────────
    eodhd_api_key: SecretStr = SecretStr("")
    fred_api_key: SecretStr = SecretStr("")

    # ── Crypto ───────────────────────────────────────────────────
    binance_api_key: SecretStr = SecretStr("")
    binance_api_secret: SecretStr = SecretStr("")

    # ── News & Sentiment ─────────────────────────────────────────
    finnhub_api_key: SecretStr = SecretStr("")
    cryptopanic_api_key: SecretStr = SecretStr("")

    # ── Infrastructure ───────────────────────────────────────────
    database_url: SecretStr = SecretStr("postgresql+asyncpg://atlas:atlas@localhost:5432/atlas")
    database_url_sync: SecretStr = SecretStr("postgresql://atlas:atlas@localhost:5432/atlas")
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    # ── OAuth ────────────────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: SecretStr = SecretStr("")
    github_client_id: str = ""
    github_client_secret: SecretStr = SecretStr("")

    # ── JWT & Frontend ───────────────────────────────────────────
    atlas_jwt_secret: SecretStr = SecretStr("change-me-in-production")
    atlas_jwt_expiry_hours: int = 24
    frontend_url: str = "http://localhost:3000"


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Singleton settings loader — reads .env once, reuses thereafter."""
    global _settings_instance  # noqa: PLW0603
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
