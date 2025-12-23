"""Common adapter interfaces and registry used by DeepSearch on Graph."""

from .base import (
    GraphAccessScope,
    GraphAdapterCapability,
    GraphAdapterMetadata,
    GraphDeepSearchAdapter,
)
from .registry import available_adapters, build_adapter, override_adapter, register_adapter

# Ensure built-in adapters are registered on import.
from . import hipporag as _hipporag  # noqa: F401

__all__ = [
    "GraphAccessScope",
    "GraphAdapterCapability",
    "GraphAdapterMetadata",
    "GraphDeepSearchAdapter",
    "register_adapter",
    "override_adapter",
    "build_adapter",
    "available_adapters",
]
