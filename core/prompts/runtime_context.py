"""Runtime prompt helpers for injecting global context (e.g. current date)."""
from datetime import datetime


_TODAY_PREFIX = "今天是"


def current_local_date_str() -> str:
    """Return current local date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def prepend_today_line(prompt: str) -> str:
    """Prepend a single-line date hint to the system prompt (idempotent)."""
    base = str(prompt or "").strip()
    line = f"{_TODAY_PREFIX} {current_local_date_str()}。"
    if not base:
        return line
    # Avoid duplicating if user/system already included a date hint.
    head = "\n".join(base.splitlines()[:2])
    if _TODAY_PREFIX in head or line in base:
        return base
    return f"{line}\n{base}"


__all__ = ["current_local_date_str", "prepend_today_line"]
