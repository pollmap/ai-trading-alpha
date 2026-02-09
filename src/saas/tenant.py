"""Multi-tenant management — user/organization isolation for SaaS mode.

Each tenant has:
- Unique tenant_id and API key
- Isolated portfolios, strategies, and simulation history
- Usage quotas and limits
- JWT-based authentication
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

from src.core.logging import get_logger

log = get_logger(__name__)


class TenantPlan(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


PLAN_LIMITS: dict[TenantPlan, dict[str, int]] = {
    TenantPlan.FREE: {
        "max_simulations": 3,
        "max_strategies": 5,
        "max_markets": 3,
        "max_api_calls_per_day": 100,
        "max_concurrent_sims": 1,
        "retention_days": 30,
    },
    TenantPlan.STARTER: {
        "max_simulations": 20,
        "max_strategies": 20,
        "max_markets": 5,
        "max_api_calls_per_day": 1_000,
        "max_concurrent_sims": 3,
        "retention_days": 90,
    },
    TenantPlan.PRO: {
        "max_simulations": 100,
        "max_strategies": 100,
        "max_markets": 10,
        "max_api_calls_per_day": 10_000,
        "max_concurrent_sims": 10,
        "retention_days": 365,
    },
    TenantPlan.ENTERPRISE: {
        "max_simulations": 999_999,
        "max_strategies": 999_999,
        "max_markets": 10,
        "max_api_calls_per_day": 999_999,
        "max_concurrent_sims": 50,
        "retention_days": 3650,
    },
}


@dataclass
class Tenant:
    """A tenant (user or organization) in the SaaS system."""

    tenant_id: str
    name: str
    email: str
    plan: TenantPlan = TenantPlan.FREE
    api_key: str = ""
    api_key_hash: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: datetime | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.api_key and not self.api_key_hash:
            self.api_key = self._generate_api_key()
            self.api_key_hash = self._hash_key(self.api_key)

    @staticmethod
    def _generate_api_key() -> str:
        return f"atlas_{secrets.token_urlsafe(32)}"

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def verify_api_key(self, key: str) -> bool:
        return hmac.compare_digest(self._hash_key(key), self.api_key_hash)

    @property
    def limits(self) -> dict[str, int]:
        return PLAN_LIMITS[self.plan]


class TenantManager:
    """In-memory tenant store. Replace with DB-backed implementation for production."""

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}
        self._key_index: dict[str, str] = {}  # api_key_hash -> tenant_id

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        email: str,
        plan: TenantPlan = TenantPlan.FREE,
    ) -> Tenant:
        """Create a new tenant and return it (including plaintext API key)."""
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            email=email,
            plan=plan,
        )
        self._tenants[tenant_id] = tenant
        self._key_index[tenant.api_key_hash] = tenant_id

        log.info(
            "tenant_created",
            tenant_id=tenant_id,
            name=name,
            plan=plan.value,
        )
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def authenticate_by_key(self, api_key: str) -> Tenant | None:
        """Look up a tenant by API key."""
        key_hash = Tenant._hash_key(api_key)
        tenant_id = self._key_index.get(key_hash)
        if tenant_id is None:
            log.warning("auth_failed_unknown_key")
            return None

        tenant = self._tenants.get(tenant_id)
        if tenant and not tenant.is_active:
            log.warning("auth_failed_inactive", tenant_id=tenant_id)
            return None

        if tenant:
            tenant.last_login = datetime.now(timezone.utc)
            log.debug("auth_success", tenant_id=tenant_id)
        return tenant

    def rotate_api_key(self, tenant_id: str) -> str | None:
        """Generate a new API key for the tenant. Returns new key or None."""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return None

        # Remove old key from index
        if tenant.api_key_hash in self._key_index:
            del self._key_index[tenant.api_key_hash]

        # Generate new key
        new_key = Tenant._generate_api_key()
        tenant.api_key = new_key
        tenant.api_key_hash = Tenant._hash_key(new_key)
        self._key_index[tenant.api_key_hash] = tenant_id

        log.info("api_key_rotated", tenant_id=tenant_id)
        return new_key

    def update_plan(self, tenant_id: str, plan: TenantPlan) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        old_plan = tenant.plan
        tenant.plan = plan
        log.info("plan_updated", tenant_id=tenant_id, old=old_plan.value, new=plan.value)
        return True

    def deactivate(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        tenant.is_active = False
        log.info("tenant_deactivated", tenant_id=tenant_id)
        return True

    def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.is_active]
        return tenants


# ── JWT Utilities ──────────────────────────────────────────────────


class JWTManager:
    """Minimal JWT implementation (HS256) — no external dependency.

    For production, use PyJWT or similar. This is a self-contained
    implementation suitable for development and small deployments.
    """

    def __init__(self, secret: str, expiry_hours: int = 24) -> None:
        self._secret: str = secret
        self._expiry_hours: int = expiry_hours

    def create_token(self, tenant_id: str, extra_claims: dict[str, str] | None = None) -> str:
        """Create a signed JWT token."""
        now = int(time.time())
        payload = {
            "sub": tenant_id,
            "iat": now,
            "exp": now + self._expiry_hours * 3600,
        }
        if extra_claims:
            payload.update(extra_claims)

        header = self._b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        body = self._b64url_encode(json.dumps(payload).encode())
        signature = self._sign(f"{header}.{body}")

        return f"{header}.{body}.{signature}"

    def verify_token(self, token: str) -> dict[str, str] | None:
        """Verify a JWT token and return the payload, or None if invalid."""
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, body_b64, sig = parts
        expected_sig = self._sign(f"{header_b64}.{body_b64}")

        if not hmac.compare_digest(sig, expected_sig):
            log.warning("jwt_invalid_signature")
            return None

        try:
            payload = json.loads(self._b64url_decode(body_b64))
        except (json.JSONDecodeError, ValueError):
            log.warning("jwt_decode_error")
            return None

        # Check expiry
        exp = payload.get("exp", 0)
        if int(time.time()) > exp:
            log.debug("jwt_expired", sub=payload.get("sub"))
            return None

        return payload

    def _sign(self, message: str) -> str:
        sig_bytes = hmac.new(
            self._secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).digest()
        return self._b64url_encode(sig_bytes)

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        import base64
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    @staticmethod
    def _b64url_decode(s: str) -> bytes:
        import base64
        padding = 4 - len(s) % 4
        if padding != 4:
            s += "=" * padding
        return base64.urlsafe_b64decode(s)
