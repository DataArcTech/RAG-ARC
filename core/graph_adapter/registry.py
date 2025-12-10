"""Registry for GraphDeepSearchAdapter implementations."""
from typing import Dict

from core.graph_adapter.base import (
    AdapterNotRegisteredError,
    GraphAdapterFactory,
    GraphDeepSearchAdapter,
)

_REGISTRY: Dict[str, GraphAdapterFactory] = {}


def register_adapter(name: str, factory: GraphAdapterFactory) -> None:
    """Register a new adapter factory."""

    if name in _REGISTRY:
        raise ValueError(f"adapter {name} already registered")
    _REGISTRY[name] = factory


def override_adapter(name: str, factory: GraphAdapterFactory) -> None:
    """Override or inject adapter implementations for tests or custom deployments."""

    _REGISTRY[name] = factory


def build_adapter(name: str, **kwargs) -> GraphDeepSearchAdapter:
    """Instantiate an adapter by name."""

    if name not in _REGISTRY:
        raise AdapterNotRegisteredError(f"adapter {name} not registered")
    return _REGISTRY[name](**kwargs)


def available_adapters() -> Dict[str, GraphAdapterFactory]:
    """Return shallow copy of the registry for inspection/CLI."""

    return dict(_REGISTRY)
