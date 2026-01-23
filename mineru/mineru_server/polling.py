import os
from typing import Tuple


DEFAULT_POLL_INTERVAL_S = 5
DEFAULT_POLL_TIMEOUT_S = 0


def _get_int_env(name: str, default: int, *, min_value: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(min_value, value)


def load_polling_config() -> Tuple[int, int]:
    interval_s = _get_int_env("MINERU_POLL_INTERVAL_S", DEFAULT_POLL_INTERVAL_S, min_value=1)
    timeout_s = _get_int_env("MINERU_POLL_TIMEOUT_S", DEFAULT_POLL_TIMEOUT_S, min_value=0)
    return interval_s, timeout_s
