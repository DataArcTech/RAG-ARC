"""Deterministic chunk sampler that mirrors TF-IDF style probes."""
from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema


class ChunkScanTool(GraphTool):
    """Samples high-signal chunks quickly to feed subsequent LLM tools."""

    descriptor = ToolDescriptor(
        name="graph.chunk_scan",
        channel="graph",
        description="Samples TF-IDF-style chunks to warm up reasoning context.",
        speed="fast",
        cost="low",
        strategy_tags=("chunk", "tfidf", "fast"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.chunk_scan",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {
                    "type": "string",
                    "description": "Optional query override when planner already decomposed the question.",
                }
            }
        ),
        example_args={
            "question": "Explain HippoRAG",
            "plan_step": "plan_01",
            "extra": {"focus_query": "HippoRAG pipeline"},
        },
    )

    def __init__(self, *, max_chunks: int = 5):
        self.max_chunks = max_chunks

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        query = self._resolve_query(request)
        payload = await adapter.aquery_subgraph(
            query,
            channel="graph",
            access_scope=request.access_scope,
        )
        chunks = self._normalize_chunks(payload.get("chunks"))
        evidences = self._to_evidences(chunks[: self.max_chunks], adapter.metadata().adapter_name)
        if not evidences:
            return ToolResult(
                summary="Chunk scan completed but no high-signal chunks surfaced.",
                diagnostics={"query": query},
            )
        summary = f"Chunk scan surfaced {len(evidences)} candidates for downstream reasoning."
        diagnostics = {
            "query": query,
            "available_chunks": len(chunks),
        }
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("ChunkScanTool requires a GraphDeepSearchAdapter instance")
        return adapter

    @staticmethod
    def _resolve_query(request: ToolRunRequest) -> str:
        if isinstance(request.extra.get("focus_query"), str):
            return request.extra["focus_query"]
        return request.question

    @staticmethod
    def _normalize_chunks(chunks: Any) -> List[Dict[str, Any]]:
        if not isinstance(chunks, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for chunk in chunks:
            if isinstance(chunk, dict) and chunk.get("content"):
                normalized.append(chunk)
        return normalized

    def _to_evidences(self, chunks: List[Dict[str, Any]], source: str) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = self._extract_chunk_id(chunk, idx)
            evidences.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=source,
                    content=str(chunk.get("content")),
                    score=self._extract_score(chunk),
                    provenance={
                        "metadata": chunk.get("metadata", {}),
                        "raw_chunk": chunk,
                        "rank": idx,
                    },
                )
            )
        return evidences

    @staticmethod
    def _extract_chunk_id(chunk: Dict[str, Any], idx: int) -> str:
        metadata = chunk.get("metadata") or {}
        return str(
            chunk.get("chunk_id")
            or chunk.get("id")
            or metadata.get("chunk_id")
            or metadata.get("id")
            or f"chunk-scan-{idx}"
        )

    @staticmethod
    def _extract_score(chunk: Dict[str, Any]) -> float | None:
        score = chunk.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return None
