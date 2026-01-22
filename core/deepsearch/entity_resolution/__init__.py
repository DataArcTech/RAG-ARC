"""Online entity/name resolution utilities for DeepSearch graph tools.

This module is intentionally framework-agnostic and avoids hidden global state.
Tools can use it to resolve noisy tool-provided strings (aliases, suffixes,
abbreviations) into canonical graph nodes in a conservative, observable way.
"""

from .factory import build_default_entity_resolver
from .resolver import EntityResolver, EntityResolutionCandidate, EntityResolutionResult

__all__ = [
    "EntityResolver",
    "EntityResolutionCandidate",
    "EntityResolutionResult",
    "build_default_entity_resolver",
]
