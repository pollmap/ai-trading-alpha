"""DB-backed tenant repository — replaces in-memory TenantManager."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from src.core.logging import get_logger
from src.saas.tenant import Tenant, TenantPlan

log = get_logger(__name__)


class TenantRepository:
    """Async PostgreSQL-backed tenant storage."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def find_by_id(self, tenant_id: str) -> Tenant | None:
        """Look up a tenant by ID."""
        async with self._engine.begin() as conn:
            row = await conn.execute(
                text("SELECT * FROM tenants WHERE tenant_id = :tid"),
                {"tid": tenant_id},
            )
            r = row.mappings().first()
            if r is None:
                return None
            return self._row_to_tenant(r)

    async def find_by_email(self, email: str) -> Tenant | None:
        """Look up a tenant by email."""
        async with self._engine.begin() as conn:
            row = await conn.execute(
                text("SELECT * FROM tenants WHERE email = :email"),
                {"email": email},
            )
            r = row.mappings().first()
            if r is None:
                return None
            return self._row_to_tenant(r)

    async def find_by_provider(
        self, provider: str, provider_id: str
    ) -> Tenant | None:
        """Look up a tenant by OAuth provider + provider_id."""
        async with self._engine.begin() as conn:
            row = await conn.execute(
                text(
                    "SELECT * FROM tenants "
                    "WHERE provider = :provider AND provider_id = :pid"
                ),
                {"provider": provider, "pid": provider_id},
            )
            r = row.mappings().first()
            if r is None:
                return None
            return self._row_to_tenant(r)

    async def upsert_oauth_user(
        self,
        *,
        tenant_id: str,
        name: str,
        email: str,
        provider: str,
        provider_id: str,
        avatar_url: str = "",
    ) -> Tenant:
        """Create or update a user from OAuth callback."""
        now = datetime.now(timezone.utc)
        api_key_hash = Tenant._hash_key(tenant_id)  # placeholder hash

        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO tenants
                        (tenant_id, name, email, plan, api_key_hash,
                         is_active, created_at, last_login,
                         provider, provider_id, avatar_url, metadata)
                    VALUES
                        (:tid, :name, :email, 'free', :hash,
                         true, :now, :now,
                         :provider, :pid, :avatar, '{}')
                    ON CONFLICT (tenant_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        avatar_url = EXCLUDED.avatar_url,
                        last_login = EXCLUDED.last_login
                    """
                ),
                {
                    "tid": tenant_id,
                    "name": name,
                    "email": email,
                    "hash": api_key_hash,
                    "now": now,
                    "provider": provider,
                    "pid": provider_id,
                    "avatar": avatar_url,
                },
            )

        log.info(
            "tenant_upserted",
            tenant_id=tenant_id,
            provider=provider,
            email=email,
        )
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            email=email,
            api_key_hash=api_key_hash,
        )
        tenant.metadata["provider"] = provider
        tenant.metadata["provider_id"] = provider_id
        tenant.metadata["avatar_url"] = avatar_url
        return tenant

    async def update_last_login(self, tenant_id: str) -> None:
        """Update the last_login timestamp."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE tenants SET last_login = :now WHERE tenant_id = :tid"
                ),
                {"now": datetime.now(timezone.utc), "tid": tenant_id},
            )

    @staticmethod
    def _row_to_tenant(r: object) -> Tenant:
        """Convert a DB row mapping to a Tenant dataclass."""
        meta = r["metadata"] if r["metadata"] else {}  # type: ignore[index]
        if isinstance(meta, str):
            meta = json.loads(meta)

        plan_str: str = r["plan"]  # type: ignore[index]
        try:
            plan = TenantPlan(plan_str)
        except ValueError:
            plan = TenantPlan.FREE

        tenant = Tenant(
            tenant_id=r["tenant_id"],  # type: ignore[index]
            name=r["name"],  # type: ignore[index]
            email=r["email"],  # type: ignore[index]
            plan=plan,
            api_key_hash=r["api_key_hash"],  # type: ignore[index]
            is_active=r["is_active"],  # type: ignore[index]
            created_at=r["created_at"],  # type: ignore[index]
            last_login=r.get("last_login"),  # type: ignore[union-attr]
            metadata=meta,
        )
        # Attach OAuth fields from metadata or columns
        if hasattr(r, "__getitem__"):
            for col in ("provider", "provider_id", "avatar_url"):
                val = r.get(col)  # type: ignore[union-attr]
                if val:
                    tenant.metadata[col] = val
        return tenant
