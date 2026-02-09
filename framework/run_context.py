"""Run-scoped context utilities.

Why this lives in framework/:
- It is infrastructure wiring, not domain logic.
- DeepSearch and other pipelines may need a stable run_id for debug dumps/metrics
  without depending on application-layer objects.

Notes:
- This is best-effort context used for observability only.
- Use ContextVar so concurrent runs do not leak metadata across tasks.
"""
import contextvars

_RUN_ID_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar("ragarc_run_id", default=None)


def set_run_id(run_id: str | None) -> contextvars.Token:
    token = (str(run_id or "").strip()) or None
    return _RUN_ID_CTX.set(token)


def reset_run_id(token: contextvars.Token) -> None:
    _RUN_ID_CTX.reset(token)


def get_run_id(*, fallback: str | None = None) -> str | None:
    value = _RUN_ID_CTX.get()
    if value:
        return str(value)
    token = (str(fallback or "").strip()) or None
    return token


__all__ = ["set_run_id", "reset_run_id", "get_run_id"]

