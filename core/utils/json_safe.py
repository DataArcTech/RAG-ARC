"""Utilities for converting objects into JSON-serializable primitives."""
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def json_safe(value: Any, *, _seen: set[int] | None = None, _depth: int = 0, _max_depth: int = 30) -> Any:
    """Return a JSON-friendly representation of `value`.

    This is intentionally conservative: it preserves primitive types and recursively
    converts common containers, dataclasses, and Pydantic models. Unknown objects
    fall back to `str(value)`.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if _depth >= _max_depth:
        return str(value)

    if _seen is None:
        _seen = set()
    obj_id = id(value)
    if obj_id in _seen:
        return "<recursion>"
    _seen.add(obj_id)

    try:
        if isinstance(value, (list, tuple, set, frozenset)):
            return [json_safe(item, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth) for item in value]

        if isinstance(value, dict):
            return {
                str(key): json_safe(val, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
                for key, val in value.items()
            }

        if isinstance(value, Path):
            return str(value)

        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
            except TypeError:
                dumped = value.model_dump(exclude_none=True)
            return json_safe(dumped, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)

        if is_dataclass(value):
            return json_safe(asdict(value), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)

        if hasattr(value, "__dict__"):
            return json_safe(vars(value), _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
    finally:
        _seen.discard(obj_id)

    return str(value)

