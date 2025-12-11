"""Tool that compares multiple adapters for determinism diagnostics."""
from collections import Counter
from typing import Dict, Iterable, List, Sequence

from core.graph_adapter.base import GraphAdapterMetadata

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema


class MultiAdapterComparatorTool(GraphTool):
    """Produces determinism diagnostics when multiple adapters are available."""

    descriptor = ToolDescriptor(
        name="graph.multi_adapter_compare",
        channel="graph",
        description="Compares adapter metadata and reports determinism ratios.",
        speed="medium",
        cost="medium",
        strategy_tags=("governance", "comparison", "adapter"),
        profile="X",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.x.multi_adapter_compare",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "alternate_adapters": {
                    "type": "array",
                    "description": "Adapter metadata provided by planner/external services.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "adapter_name": {"type": "string"},
                            "graph_type": {"type": "string"},
                            "version": {"type": "string"},
                            "domain_tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["adapter_name", "graph_type", "version"],
                    },
                }
            }
        ),
        example_args={
            "question": "Compare adapters",
            "plan_step": "plan_meta",
            "extra": {
                "alternate_adapters": [
                    {"adapter_name": "hipporag", "graph_type": "hipporag", "version": "1.0"},
                    {"adapter_name": "lightrag", "graph_type": "lightrag", "version": "latest"},
                ]
            },
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapters = self._collect_adapters(request)
        if not adapters:
            return ToolResult(
                summary="Multi-adapter comparison skipped because only one adapter is available.",
                diagnostics={"adapter_count": 0},
            )
        ratio = self._determinism_ratio(adapters)
        summary = f"Multi-adapter comparison completed with determinism ratio {ratio:.2f}."
        diagnostics = {
            "adapter_count": len(adapters),
            "determinism_ratio": ratio,
            "graph_types": [meta.graph_type for meta in adapters],
        }
        return ToolResult(summary=summary, diagnostics=diagnostics)

    def _collect_adapters(self, request: ToolRunRequest) -> List[GraphAdapterMetadata]:
        adapters: List[GraphAdapterMetadata] = []
        if request.adapter:
            adapters.append(request.adapter.metadata())
        for payload in request.extra.get("alternate_adapters", []):
            meta = self._coerce_metadata(payload)
            if meta:
                adapters.append(meta)
        return adapters

    @staticmethod
    def _coerce_metadata(payload) -> GraphAdapterMetadata | None:
        if isinstance(payload, GraphAdapterMetadata):
            return payload
        if isinstance(payload, dict):
            required = {"adapter_name", "graph_type", "version"}
            if required.issubset(payload.keys()):
                return GraphAdapterMetadata(
                    adapter_name=str(payload["adapter_name"]),
                    graph_type=str(payload["graph_type"]),
                    version=str(payload["version"]),
                    owner=payload.get("owner"),
                    domain_tags=tuple(payload.get("domain_tags", []) or []),
                )
        return None

    @staticmethod
    def _determinism_ratio(adapters: Sequence[GraphAdapterMetadata]) -> float:
        counts = Counter(meta.graph_type for meta in adapters)
        if not counts:
            return 0.0
        consensus = counts.most_common(1)[0][1]
        return consensus / len(adapters)
