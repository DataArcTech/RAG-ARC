"""Tenant context utilities.

Why this lives in framework/:
- It is infrastructure wiring, not domain logic.
- Both core/ and encapsulation/ may need tenant context without depending on each other.

Notes:
- In current deployments, a single service usually serves a single tenant; tenant_id defaults to "default".
- In future multi-tenant deployments, tenant_id should be set by auth/session middleware (e.g., from JWT claims).
"""
import contextvars
import os

_TENANT_ID_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar("ragarc_tenant_id", default=None)


def default_tenant_id() -> str:
    return (os.getenv("JWT_DEFAULT_TENANT_ID") or os.getenv("RAGARC_DEFAULT_TENANT_ID") or "default").strip() or "default"


def set_tenant_id(tenant_id: str | None) -> None:
    token = (str(tenant_id or "").strip()) or None
    _TENANT_ID_CTX.set(token)


def get_tenant_id(*, fallback: str | None = None) -> str:
    value = _TENANT_ID_CTX.get()
    if value:
        return str(value)
    return str(fallback or default_tenant_id())

