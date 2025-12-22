"""Graph adapter protocol powering DeepSearch on Graph."""
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class GraphAccessScope:
    """Represents the logical scope that limits adapter visibility (tenant/user/etc)."""

    scope_id: str
    scope_type: str = "owner"
    labels: Sequence[str] = field(default_factory=tuple)
    attributes: Mapping[str, Any] | None = None

    def as_token(self) -> str:
        """Return the opaque token used by downstream graph stores."""

        return self.scope_id


@dataclass
class GraphAdapterCapability:
    """How an adapter reports its supported modes and metrics."""

    name: str
    modes: Sequence[str] = field(default_factory=tuple)
    metrics: Mapping[str, Any] | None = None
    extra: Mapping[str, Any] | None = None


@dataclass
class GraphAdapterMetadata:
    """Metadata published by adapters for monitoring and auditing."""

    adapter_name: str
    graph_type: str
    version: str
    owner: Optional[str] = None
    capabilities: Sequence[GraphAdapterCapability] = field(default_factory=tuple)
    domain_tags: Sequence[str] = field(default_factory=tuple)
    config_fingerprint: Optional[str] = None


@runtime_checkable
class GraphDeepSearchAdapter(Protocol):
    """Adapter protocol exposing the operations used by DeepSearch."""

    async def prepare(self, question: str, *, access_scope: Optional[GraphAccessScope] = None) -> None:
        """Warm up caches or load graph shards before traversal."""

    async def aquery_subgraph(
        self,
        query: str,
        *,
        channel: str = "graph",
        access_scope: Optional[GraphAccessScope] = None,
    ) -> Dict[str, Any]:
        """Return a subgraph result for the current query/plan step."""

    async def context_filter(
        self,
        data: Mapping[str, Any],
        *,
        filter_type: str = "semantic",
        access_scope: Optional[GraphAccessScope] = None,
    ) -> Mapping[str, Any]:
        """Apply semantic/relational filters to the retrieved subgraph."""

    async def summarize(
        self,
        channel: str,
        data: Mapping[str, Any],
        *,
        access_scope: Optional[GraphAccessScope] = None,
    ) -> str:
        """Return a textual summary consumable by downstream LLMs."""

    async def chain_traverse(
        self,
        strategy: Mapping[str, Any],
        *,
        access_scope: Optional[GraphAccessScope] = None,
    ) -> Mapping[str, Any]:
        """Run a chain-of-exploration strategy using adapter-specific primitives.

        Implementations should return a mapping that is safe to JSON-serialize.
        Minimum recommended keys for observability and tool contracts:
        - strategy: str
        - hops: int
        - visited: list[str] (best-effort)
        Optional keys depending on strategy:
        - paths: list[dict] (e.g. for "ppr_prefetch")
        - bridges: list[dict] (e.g. for "bridge_lookup")
        """

    def metadata(self) -> GraphAdapterMetadata:
        """Return adapter metadata for telemetry and feature gating."""


class AdapterNotRegisteredError(RuntimeError):
    """Raised when the registry cannot resolve an adapter name."""


class GraphAdapterFactory(Protocol):
    """Factory signature used by the registry."""

    def __call__(self, **kwargs: Any) -> GraphDeepSearchAdapter:
        """Return a ready-to-use GraphDeepSearchAdapter instance."""
