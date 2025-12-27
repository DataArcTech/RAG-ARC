"""Hybrid tool combining deterministic probes with LLM summaries."""
from typing import Any, Dict, List

from config.core.deepsearch.tool_defaults import (
    HYBRID_NEIGHBORHOOD_DEFAULT_MAX_CHUNKS,
    HYBRID_NEIGHBORHOOD_DEFAULT_PATTERN_MAX_TERMS,
    HYBRID_NEIGHBORHOOD_DEFAULT_SNIPPET_CHARS,
    HYBRID_NEIGHBORHOOD_DEFAULT_SUMMARY_TEMPERATURE,
    HYBRID_NEIGHBORHOOD_DEFAULT_TRAVERSAL_MAX_DEPTH,
    HYBRID_NEIGHBORHOOD_DEFAULT_TRAVERSAL_STRATEGY,
)
from encapsulation.data_model.deepsearch import EvidenceChunk

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async
from ..fast.pattern_probe import PatternProbeTool
from core.graph_adapter.concurrency import adapter_locked
from core.prompts.deepsearch import HYBRID_NEIGHBORHOOD_SUMMARY_PROMPT


class HybridNeighborhoodProbeTool(GraphTool):
    """Balances grep-style probes with LLM synthesis for better recall/precision."""

    descriptor = ToolDescriptor(
        name="graph.hybrid_neighborhood",
        channel="graph",
        description="Hybrid GraphRAG probe that mixes deterministic chunk filters with LLM summarisation.",
        speed="medium",
        cost="medium",
        strategy_tags=("hybrid", "graphrag", "llm", "chunk_triple"),
        profile="X",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.x.hybrid_neighborhood",
        mcp_callable=True,
        example_args={
            "question": "Analyze OpenAI history",
            "plan_step": "plan_01",
            "context_evidences": [],
        },
    )

    def __init__(
        self,
        llm_connector,
        *,
        max_chunks: int = HYBRID_NEIGHBORHOOD_DEFAULT_MAX_CHUNKS,
        pattern_probe_max_terms: int = HYBRID_NEIGHBORHOOD_DEFAULT_PATTERN_MAX_TERMS,
        traversal_strategy: str = HYBRID_NEIGHBORHOOD_DEFAULT_TRAVERSAL_STRATEGY,
        traversal_max_depth: int = HYBRID_NEIGHBORHOOD_DEFAULT_TRAVERSAL_MAX_DEPTH,
        snippet_chars: int = HYBRID_NEIGHBORHOOD_DEFAULT_SNIPPET_CHARS,
        temperature: float = HYBRID_NEIGHBORHOOD_DEFAULT_SUMMARY_TEMPERATURE,
    ):
        self.llm_connector = llm_connector
        self.max_chunks = max_chunks
        self._pattern_tool = PatternProbeTool(max_terms=int(pattern_probe_max_terms))
        self.traversal_strategy = str(traversal_strategy).strip()
        if not self.traversal_strategy:
            raise ValueError("HybridNeighborhoodProbeTool traversal_strategy must be a non-empty string")
        self.traversal_max_depth = int(traversal_max_depth)
        self.snippet_chars = int(snippet_chars)
        self.temperature = float(temperature)

    async def run(self, request: ToolRunRequest) -> ToolResult:
        # Step 1: deterministic scan to surface candidate chunks quickly.
        pattern_result = await self._pattern_tool.run(request)
        if not pattern_result.evidences:
            return ToolResult(
                summary="Hybrid probe could not surface candidates via rule-based scan.",
                diagnostics={"pattern_summary": pattern_result.summary},
            )

        # Step 2: fetch richer neighborhoods for the strongest matches.
        adapter = request.adapter
        if adapter is None:
            raise RuntimeError("HybridNeighborhoodProbeTool requires a GraphDeepSearchAdapter")

        enriched: List[EvidenceChunk] = []
        async with adapter_locked(adapter):
            for ev in pattern_result.evidences[: self.max_chunks]:
                subgraph = await adapter.chain_traverse(
                    {
                        "strategy": self.traversal_strategy,
                        "seed_chunk": ev.chunk_id,
                        "max_depth": self.traversal_max_depth,
                    },
                    access_scope=request.access_scope,
                )
                enriched.append(
                    EvidenceChunk(
                        chunk_id=f"{ev.chunk_id}-hybrid",
                        source=adapter.metadata().adapter_name,
                        content=ev.content,
                        score=ev.score,
                        provenance={
                            "pattern_match": ev.provenance,
                            "chain_traverse": subgraph,
                        },
                    )
                )

        # Step 3: summarise with LLM for explainability.
        summary = await self._summarize(request, enriched)
        determinism_ratio = len(pattern_result.evidences) / max(1, len(enriched))
        diagnostics: Dict[str, Any] = {
            "pattern_summary": pattern_result.summary,
            "enriched_count": len(enriched),
            "determinism_ratio": min(1.0, determinism_ratio),
        }
        diagnostics["token_breakdown"] = self._token_breakdown(pattern_result.evidences, summary)
        return ToolResult(summary=summary, evidences=enriched, diagnostics=diagnostics)

    async def _summarize(self, request: ToolRunRequest, evidences: List[EvidenceChunk]) -> str:
        context = "\n\n".join(ev.content[: self.snippet_chars] for ev in evidences)
        messages = [
            {
                "role": "system",
                "content": HYBRID_NEIGHBORHOOD_SUMMARY_PROMPT,
            },
            {
                "role": "user",
                "content": f"Question: {request.question}\n\nContext:\n{context}",
            },
        ]
        response = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
        rendered = (response or "").strip()
        if not rendered:
            raise RuntimeError("HybridNeighborhoodProbeTool returned an empty response")
        return rendered

    @staticmethod
    def _token_breakdown(evidences: List[EvidenceChunk], summary_text: str) -> Dict[str, int]:
        deterministic_tokens = sum(len(ev.content.split()) for ev in evidences)
        llm_tokens = len(summary_text.split()) if summary_text else 0
        return {
            "deterministic_tokens": deterministic_tokens,
            "llm_tokens": llm_tokens,
        }
