"""Framework-level cache utilities.

Keep caching primitives out of `core/` so core logic stays framework-agnostic.
"""

from .ttl_lru import TTLRUCache

__all__ = ["TTLRUCache"]

