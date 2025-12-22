"""Bridge lookup tool that highlights entity-triple pivots."""
from typing import Any, Dict, Iterable, List

from encapsulation.data_model.deepsearch import EvidenceChunk

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from core.graph_adapter.concurrency import adapter_locked


class BridgeLookupTool(GraphTool):
    """Fetches bridge triples that connect disjoint reasoning branches."""

    descriptor = ToolDescriptor(
        name="graph.bridge_lookup",
        channel="graph",
        description="Deterministic entity→triple lookup used to stitch reasoning hops.",
        speed="fast",
        cost="low",
        strategy_tags=("bridge", "triple", "entity"),
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
        evidences = self._build_evidences(bridges[: self.max_results], adapter.metadata().adapter_name)
        summary = f"Bridge lookup highlighted {len(evidences)} connective triples."
        diagnostics = {
            "available_bridges": len(bridges),
            "strategy": traversal.get("strategy"),
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

    def _build_evidences(self, bridges: Iterable[Dict[str, Any]], source: str) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        for idx, bridge in enumerate(bridges):
            head = bridge.get("head") or bridge.get("entity") or "unknown_head"
            relation = bridge.get("relation") or bridge.get("predicate") or "related_to"
            tail = bridge.get("tail") or bridge.get("target") or "unknown_tail"
            content = f"{head} -[{relation}]-> {tail}"
            chunk_id = str(
                bridge.get("id")
                or bridge.get("bridge_id")
                or bridge.get("metadata", {}).get("id")
                or f"bridge-{idx}"
            )
            evidences.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=source,
                    content=content,
                    score=bridge.get("score"),
                    provenance={
                        "relation": relation,
                        "raw": bridge,
                    },
                )
            )
        return evidences
