"""Tests for TenantRepository — DB-backed tenant storage."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.db.tenants import TenantRepository
from src.saas.tenant import Tenant, TenantPlan


class _FakeMapping(dict):  # type: ignore[type-arg]
    """Dict subclass that also supports attribute access for .get()."""
    pass


def _make_row(**kwargs: object) -> _FakeMapping:
    """Create a fake DB row mapping."""
    defaults = {
        "tenant_id": "t1",
        "name": "Test User",
        "email": "test@example.com",
        "plan": "free",
        "api_key_hash": "abc123",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
        "last_login": None,
        "provider": "google",
        "provider_id": "g-123",
        "avatar_url": "https://img.example.com/test.jpg",
        "metadata": {},
    }
    defaults.update(kwargs)
    return _FakeMapping(defaults)


def _mock_engine(mock_conn: AsyncMock) -> MagicMock:
    """Create a mock engine with proper async context manager for begin()."""
    engine = MagicMock()

    @asynccontextmanager
    async def _begin() -> AsyncIterator[AsyncMock]:
        yield mock_conn

    engine.begin = _begin
    return engine


class TestRowToTenant:
    def test_basic_conversion(self) -> None:
        row = _make_row()
        tenant = TenantRepository._row_to_tenant(row)
        assert tenant.tenant_id == "t1"
        assert tenant.name == "Test User"
        assert tenant.email == "test@example.com"
        assert tenant.plan == TenantPlan.FREE
        assert tenant.is_active is True

    def test_pro_plan(self) -> None:
        row = _make_row(plan="pro")
        tenant = TenantRepository._row_to_tenant(row)
        assert tenant.plan == TenantPlan.PRO

    def test_unknown_plan_defaults_free(self) -> None:
        row = _make_row(plan="nonexistent")
        tenant = TenantRepository._row_to_tenant(row)
        assert tenant.plan == TenantPlan.FREE

    def test_metadata_as_string(self) -> None:
        row = _make_row(metadata='{"custom": "value"}')
        tenant = TenantRepository._row_to_tenant(row)
        assert tenant.metadata.get("custom") == "value"

    def test_oauth_fields_in_metadata(self) -> None:
        row = _make_row(provider="github", provider_id="gh-99", avatar_url="https://img.github.com/x.jpg")
        tenant = TenantRepository._row_to_tenant(row)
        assert tenant.metadata["provider"] == "github"
        assert tenant.metadata["provider_id"] == "gh-99"
        assert tenant.metadata["avatar_url"] == "https://img.github.com/x.jpg"

    def test_none_metadata(self) -> None:
        row = _make_row(metadata=None)
        tenant = TenantRepository._row_to_tenant(row)
        assert isinstance(tenant.metadata, dict)


class TestFindById:
    @pytest.mark.asyncio
    async def test_found(self) -> None:
        row = _make_row(tenant_id="found-id")
        mock_result = MagicMock()
        mock_result.mappings.return_value.first.return_value = row

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        engine = _mock_engine(mock_conn)
        repo = TenantRepository(engine)
        tenant = await repo.find_by_id("found-id")
        assert tenant is not None
        assert tenant.tenant_id == "found-id"

    @pytest.mark.asyncio
    async def test_not_found(self) -> None:
        mock_result = MagicMock()
        mock_result.mappings.return_value.first.return_value = None

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        engine = _mock_engine(mock_conn)
        repo = TenantRepository(engine)
        tenant = await repo.find_by_id("no-such-id")
        assert tenant is None


class TestFindByEmail:
    @pytest.mark.asyncio
    async def test_found(self) -> None:
        row = _make_row(email="found@test.com")
        mock_result = MagicMock()
        mock_result.mappings.return_value.first.return_value = row

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result

        engine = _mock_engine(mock_conn)
        repo = TenantRepository(engine)
        tenant = await repo.find_by_email("found@test.com")
        assert tenant is not None
        assert tenant.email == "found@test.com"
