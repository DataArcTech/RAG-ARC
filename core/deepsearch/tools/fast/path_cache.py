"""Path cache tool that triggers adapter-specific prefetch strategies."""
from typing import Any, Dict, Iterable, List

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_CHAIN_TRAVERSE, SCOPE_OWNER
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id


class PathCacheTool(GraphTool):
    """Caches promising paths via adapter traversal primitives."""

    descriptor = ToolDescriptor(
        name="graph.path_cache",
        channel="graph",
        description=(
            "Deterministic prefetch over adapter chain_traverse(ppr_prefetch) to warm caches. "
            "Evidence: derived paths (NOT citeable)."
        ),
        speed="fast",
        cost="medium",
        strategy_tags=("ppr", "prefetch", "cache", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_CHAIN_TRAVERSE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.path_cache",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "seed_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hint entities for personalised PageRank cache warmup.",
                }
            }
        ),
        example_args={
            "question": "Trace the relationship between company A and B",
            "plan_step": "plan_03",
            "extra": {"seed_entities": ["Company A", "Company B"]},
        },
    )

    def __init__(self, *, max_paths: int = 3):
        self.max_paths = max_paths

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not self._supports_chain_mode(adapter, "ppr_prefetch"):
            return ToolResult(
                summary="Path cache skipped because the adapter does not advertise ppr_prefetch support.",
                diagnostics={
                    "requested_strategy": "ppr_prefetch",
                    "adapter": getattr(adapter.metadata(), "adapter_name", None),
                },
            )
        async with adapter_locked(adapter):
            traversal = await adapter.chain_traverse(
                {
                    "strategy": "ppr_prefetch",
                    "question": request.question,
                    "seed_entities": request.extra.get("seed_entities") or [],
                    "max_paths": self.max_paths,
                },
                access_scope=request.access_scope,
            )
        paths = self._normalize_paths(traversal.get("paths"))
        evidences = self._paths_to_evidence(
            paths[: self.max_paths],
            source=adapter.metadata().adapter_name,
            tool_name=self.descriptor.name,
            plan_step=request.plan_step,
        )
        diagnostics = {
            "prefetch_strategy": traversal.get("strategy"),
            "prefetched_paths": len(paths),
            "hops": traversal.get("hops"),
        }
        if not evidences:
            return ToolResult(
                summary="Path cache did not report any prefetched paths.",
                diagnostics=diagnostics,
            )
        summary = f"Path cache stored {len(evidences)} candidate walks for reuse."
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("PathCacheTool requires a GraphDeepSearchAdapter instance")
        return adapter

    @staticmethod
    def _normalize_paths(paths: Any) -> List[Dict[str, Any]]:
        if isinstance(paths, list):
            return [path for path in paths if isinstance(path, dict)]
        return []

    def _paths_to_evidence(
        self,
        paths: Iterable[Dict[str, Any]],
        *,
        source: str,
        tool_name: str,
        plan_step: str | None,
    ) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        for idx, path in enumerate(paths):
            visited_nodes = path.get("nodes") or path.get("visited_nodes") or []
            content = " -> ".join(str(node) for node in visited_nodes) or str(path)
            chunk_id = derived_chunk_id(tool_name=tool_name, plan_step=plan_step, label=f"path_{idx}", content=content)
            evidences.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=tool_name,
                    content=content,
                    kind=EVIDENCE_KIND_DERIVED,
                    score=path.get("score"),
                    provenance={"raw_path": path},
                )
            )
        return evidences

    @staticmethod
    def _supports_chain_mode(adapter, mode: str) -> bool:
        try:
            metadata = adapter.metadata()
        except Exception:
            return False
        capabilities = getattr(metadata, "capabilities", None)
        if not isinstance(capabilities, (list, tuple)):
            return False
        for cap in capabilities:
            if getattr(cap, "name", None) != "chain_of_exploration":
                continue
            modes = getattr(cap, "modes", None)
            if isinstance(modes, (list, tuple, set)) and mode in modes:
                return True
        return False
