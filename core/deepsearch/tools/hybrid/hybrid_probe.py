"""Hybrid tool combining deterministic probes with LLM summaries."""
from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async
from ..fast.pattern_probe import PatternProbeTool
from core.graph_adapter.concurrency import adapter_locked


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

    def __init__(self, llm_connector, *, max_chunks: int = 5):
        self.llm_connector = llm_connector
        self.max_chunks = max_chunks
        self._pattern_tool = PatternProbeTool(max_terms=3)

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
                        "strategy": "ppr_chain",
                        "seed_chunk": ev.chunk_id,
                        "max_depth": 2,
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
        context = "\n\n".join(ev.content[:400] for ev in evidences)
        messages = [
            {
                "role": "system",
                "content": "You condense chunks on graph into concise reasoning bullets.",
            },
            {
                "role": "user",
                "content": f"Question: {request.question}\n\nContext:\n{context}",
            },
        ]
        try:
            response = await call_llm_async(self.llm_connector, messages, temperature=0.1)
            return response.strip()
        except Exception:
            return "Hybrid probe summarisation failed; returning raw chunk snippets."

    @staticmethod
    def _token_breakdown(evidences: List[EvidenceChunk], summary_text: str) -> Dict[str, int]:
        deterministic_tokens = sum(len(ev.content.split()) for ev in evidences)
        llm_tokens = len(summary_text.split()) if summary_text else 0
        return {
            "deterministic_tokens": deterministic_tokens,
            "llm_tokens": llm_tokens,
        }
