"""Tests for SaaS multi-tenant layer."""

from __future__ import annotations

import time

import pytest

from src.saas.tenant import (
    Tenant,
    TenantManager,
    TenantPlan,
    JWTManager,
    PLAN_LIMITS,
)
from src.saas.usage import UsageMeter, UsageRecord


class TestTenant:
    """Tests for Tenant dataclass."""

    def test_auto_generates_api_key(self) -> None:
        t = Tenant(tenant_id="t1", name="Test", email="test@example.com")
        assert t.api_key.startswith("atlas_")
        assert len(t.api_key) > 20
        assert t.api_key_hash != ""

    def test_verify_api_key(self) -> None:
        t = Tenant(tenant_id="t1", name="Test", email="test@example.com")
        assert t.verify_api_key(t.api_key) is True
        assert t.verify_api_key("wrong_key") is False

    def test_limits_per_plan(self) -> None:
        t = Tenant(tenant_id="t1", name="Test", email="test@example.com", plan=TenantPlan.PRO)
        limits = t.limits
        assert limits["max_simulations"] == 100
        assert limits["max_strategies"] == 100

    def test_free_plan_limits(self) -> None:
        limits = PLAN_LIMITS[TenantPlan.FREE]
        assert limits["max_simulations"] == 3
        assert limits["max_markets"] == 3


class TestTenantManager:
    """Tests for TenantManager CRUD operations."""

    def test_create_and_get(self) -> None:
        mgr = TenantManager()
        t = mgr.create_tenant("t1", "Alice", "alice@example.com")
        assert t.tenant_id == "t1"
        assert t.name == "Alice"

        retrieved = mgr.get_tenant("t1")
        assert retrieved is not None
        assert retrieved.email == "alice@example.com"

    def test_authenticate_by_key(self) -> None:
        mgr = TenantManager()
        t = mgr.create_tenant("t1", "Bob", "bob@example.com")
        api_key = t.api_key

        authed = mgr.authenticate_by_key(api_key)
        assert authed is not None
        assert authed.tenant_id == "t1"
        assert authed.last_login is not None

    def test_authenticate_wrong_key(self) -> None:
        mgr = TenantManager()
        mgr.create_tenant("t1", "Bob", "bob@example.com")
        assert mgr.authenticate_by_key("bad_key") is None

    def test_authenticate_inactive(self) -> None:
        mgr = TenantManager()
        t = mgr.create_tenant("t1", "Charlie", "charlie@example.com")
        mgr.deactivate("t1")
        assert mgr.authenticate_by_key(t.api_key) is None

    def test_rotate_api_key(self) -> None:
        mgr = TenantManager()
        t = mgr.create_tenant("t1", "Dave", "dave@example.com")
        old_key = t.api_key

        new_key = mgr.rotate_api_key("t1")
        assert new_key is not None
        assert new_key != old_key

        # Old key should no longer work
        assert mgr.authenticate_by_key(old_key) is None
        # New key should work
        assert mgr.authenticate_by_key(new_key) is not None

    def test_update_plan(self) -> None:
        mgr = TenantManager()
        mgr.create_tenant("t1", "Eve", "eve@example.com")
        assert mgr.update_plan("t1", TenantPlan.PRO) is True

        t = mgr.get_tenant("t1")
        assert t is not None
        assert t.plan == TenantPlan.PRO

    def test_list_tenants(self) -> None:
        mgr = TenantManager()
        mgr.create_tenant("t1", "A", "a@example.com")
        mgr.create_tenant("t2", "B", "b@example.com")
        mgr.deactivate("t2")

        active = mgr.list_tenants(active_only=True)
        assert len(active) == 1

        all_tenants = mgr.list_tenants(active_only=False)
        assert len(all_tenants) == 2


class TestJWTManager:
    """Tests for JWT token creation and verification."""

    def test_create_and_verify(self) -> None:
        jwt = JWTManager(secret="test-secret", expiry_hours=1)
        token = jwt.create_token("tenant-123")

        payload = jwt.verify_token(token)
        assert payload is not None
        assert payload["sub"] == "tenant-123"

    def test_expired_token(self) -> None:
        jwt = JWTManager(secret="test-secret", expiry_hours=0)
        # Create a token that's already expired
        token = jwt.create_token("tenant-123")
        # It should be expired since expiry_hours=0 means exp = iat
        time.sleep(1)
        payload = jwt.verify_token(token)
        assert payload is None

    def test_invalid_signature(self) -> None:
        jwt1 = JWTManager(secret="secret-1", expiry_hours=1)
        jwt2 = JWTManager(secret="secret-2", expiry_hours=1)

        token = jwt1.create_token("tenant-123")
        payload = jwt2.verify_token(token)
        assert payload is None

    def test_malformed_token(self) -> None:
        jwt = JWTManager(secret="test-secret")
        assert jwt.verify_token("not.a.valid.token.format") is None
        assert jwt.verify_token("") is None
        assert jwt.verify_token("abc") is None

    def test_extra_claims(self) -> None:
        jwt = JWTManager(secret="test-secret", expiry_hours=1)
        token = jwt.create_token("t1", extra_claims={"role": "admin"})
        payload = jwt.verify_token(token)
        assert payload is not None
        assert payload["role"] == "admin"


class TestUsageMeter:
    """Tests for usage metering."""

    def test_record_and_summary(self) -> None:
        meter = UsageMeter()
        meter.record(UsageRecord(tenant_id="t1", event_type="api_call"))
        meter.record(UsageRecord(tenant_id="t1", event_type="api_call"))
        meter.record(UsageRecord(tenant_id="t1", event_type="llm_call"))

        summary = meter.get_daily_summary("t1")
        assert summary.api_calls == 2
        assert summary.llm_calls == 1

    def test_check_quota_within_limits(self) -> None:
        meter = UsageMeter()
        tenant = Tenant(tenant_id="t1", name="Test", email="test@example.com", plan=TenantPlan.FREE)

        # Free plan allows 100 API calls per day
        assert meter.check_quota(tenant, "api_call") is True

    def test_check_quota_exceeded(self) -> None:
        meter = UsageMeter()
        tenant = Tenant(tenant_id="t1", name="Test", email="test@example.com", plan=TenantPlan.FREE)

        # Record 100 calls (Free plan limit)
        for _ in range(100):
            meter.record(UsageRecord(tenant_id="t1", event_type="api_call"))

        assert meter.check_quota(tenant, "api_call") is False

    def test_monthly_summary(self) -> None:
        meter = UsageMeter()
        meter.record(UsageRecord(tenant_id="t1", event_type="api_call"))

        summary = meter.get_monthly_summary("t1")
        assert summary.api_calls == 1

    def test_get_records(self) -> None:
        meter = UsageMeter()
        for i in range(5):
            meter.record(UsageRecord(
                tenant_id="t1",
                event_type="api_call",
                metadata={"request": str(i)},
            ))

        records = meter.get_all_records("t1")
        assert len(records) == 5

    def test_reset_daily(self) -> None:
        meter = UsageMeter()
        meter.record(UsageRecord(tenant_id="t1", event_type="api_call"))
        meter.reset_daily("t1")
        summary = meter.get_daily_summary("t1")
        assert summary.api_calls == 0
