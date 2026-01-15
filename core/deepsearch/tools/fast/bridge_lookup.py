"""Bridge lookup tool that highlights entity-triple pivots."""
from typing import Any, Dict, Iterable, List

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_CHAIN_TRAVERSE, SCOPE_OWNER
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id


class BridgeLookupTool(GraphTool):
    """Fetches bridge triples that connect disjoint reasoning branches."""

    descriptor = ToolDescriptor(
        name="graph.bridge_lookup",
        channel="graph",
        description=(
            "Deterministic bridge lookup over adapter chain_traverse(bridge_lookup) results. "
            "Evidence: derived triples (NOT citeable; use chunk/cypher tools for citations)."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("bridge", "triple", "entity", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_CHAIN_TRAVERSE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.bridge_lookup",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "seed_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Seed entities or chunk IDs used to anchor the traversal.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Optional override for how many bridge triples to return.",
                    "minimum": 0,
                },
                "max_results": {
                    "type": "integer",
                    "description": "Alias of top_k for backward compatibility.",
                    "minimum": 0,
                }
            }
        ),
        example_args={
            "question": "How is OpenAI related to DeepMind?",
            "plan_step": "plan_02",
            "extra": {"seed_entities": ["OpenAI", "DeepMind"]},
        },
    )

    def __init__(self, *, max_results: int = 4):
        self.max_results = max_results

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not self._supports_chain_mode(adapter, "bridge_lookup"):
            return ToolResult(
                summary="Bridge lookup skipped because the adapter does not advertise bridge_lookup support.",
                diagnostics={
                    "requested_strategy": "bridge_lookup",
                    "adapter": getattr(adapter.metadata(), "adapter_name", None),
                },
            )
        async with adapter_locked(adapter):
            traversal = await adapter.chain_traverse(
                {
                    "strategy": "bridge_lookup",
                    "question": request.question,
                    "seed_entities": request.extra.get("seed_entities") or [],
                },
                access_scope=request.access_scope,
            )
        bridges = self._extract_bridges(traversal)
        if not bridges:
            return ToolResult(
                summary="Bridge lookup completed but no connective triples were reported.",
                diagnostics={"strategy": traversal.get("strategy")},
            )
        override = request.extra.get("top_k", None)
        if override is None:
            override = request.extra.get("max_results", None)
        try:
            limit = int(override) if override is not None else int(self.max_results)
        except Exception:
            limit = int(self.max_results) if self.max_results is not None else 0
        if limit < 0:
            limit = 0
        evidences = self._build_evidences(
            bridges[:limit] if limit else [],
            source=adapter.metadata().adapter_name,
            tool_name=self.descriptor.name,
            plan_step=request.plan_step,
        )
        summary = f"Bridge lookup highlighted {len(evidences)} connective triples."
        diagnostics = {
            "available_bridges": len(bridges),
            "strategy": traversal.get("strategy"),
            "top_k": limit,
        }
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("BridgeLookupTool requires a GraphDeepSearchAdapter instance")
        return adapter

    @staticmethod
    def _extract_bridges(traversal: Dict[str, Any]) -> List[Dict[str, Any]]:
        bridges = traversal.get("bridges")
        if isinstance(bridges, list):
            return [b for b in bridges if isinstance(b, dict)]
        chunks = traversal.get("chunks")
        if isinstance(chunks, list):
            return [c for c in chunks if isinstance(c, dict)]
        if traversal:
            return [traversal]
        return []

    def _build_evidences(
        self,
        bridges: Iterable[Dict[str, Any]],
        *,
        source: str,
        tool_name: str,
        plan_step: str | None,
    ) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        for idx, bridge in enumerate(bridges):
            head = bridge.get("head") or bridge.get("entity")
            tail = bridge.get("tail") or bridge.get("target")
            if not head or not tail:
                continue
            relation = bridge.get("relation") or bridge.get("predicate") or "related_to"
            content = f"{head} -[{relation}]-> {tail}"
            chunk_id = derived_chunk_id(tool_name=tool_name, plan_step=plan_step, label=f"bridge_{idx}", content=content)
            evidences.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=tool_name,
                    content=content,
                    kind=EVIDENCE_KIND_DERIVED,
                    score=bridge.get("score"),
                    provenance={
                        "relation": relation,
                        "triple": {"head": str(head), "relation": str(relation), "tail": str(tail)},
                        "raw": bridge,
                    },
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
