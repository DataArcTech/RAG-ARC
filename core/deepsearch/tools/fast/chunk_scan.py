"""Deterministic chunk sampler that mirrors TF-IDF style probes."""
from typing import Any, Dict, List

from config.core.deepsearch.tool_defaults import CHUNK_SCAN_DEFAULT_MAX_CHUNKS, CHUNK_SCAN_DEFAULT_QUERY_MAX_CHARS
from encapsulation.data_model.deepsearch import EvidenceChunk

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, SCOPE_FILE, SCOPE_OWNER
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.query_clean import clean_query
from core.deepsearch.utils.file_scope import chunk_in_scope, resolve_file_scope


class ChunkScanTool(GraphTool):
    """Samples high-signal chunks quickly to feed subsequent LLM tools."""

    descriptor = ToolDescriptor(
        name="graph.chunk_scan",
        channel="graph",
        description=(
            "Deterministic fast chunk sampler (TF-IDF-style) to bootstrap evidence. "
            "Evidence: primary chunks (citeable); respects file_scope when enabled."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("chunk", "tfidf", "fast", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.chunk_scan",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "focus_query": {
                    "type": "string",
                    "description": "Optional query override when planner already decomposed the question.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Optional override for how many chunks to return.",
                    "minimum": 0,
                },
                "max_chunks": {
                    "type": "integer",
                    "description": "Alias of top_k for backward compatibility.",
                    "minimum": 0,
                }
            }
        ),
        example_args={
            "question": "Explain HippoRAG",
            "plan_step": "plan_01",
            "extra": {"focus_query": "HippoRAG pipeline"},
        },
    )

    def __init__(self, *, max_chunks: int = CHUNK_SCAN_DEFAULT_MAX_CHUNKS):
        self.max_chunks = max_chunks

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        query = self._resolve_query(request)
        override = request.extra.get("top_k", None)
        if override is None:
            override = request.extra.get("max_chunks", None)
        try:
            effective_max = int(override) if override is not None else int(self.max_chunks)
        except Exception:
            effective_max = int(self.max_chunks) if self.max_chunks is not None else 0
        max_chunks = effective_max if effective_max is not None else 0
        if max_chunks < 0:
            max_chunks = 0
        file_scope = resolve_file_scope(
            extra=request.extra,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
            question=request.question,
        )
        async with adapter_locked(adapter):
            payload = await adapter.aquery_subgraph(
                query,
                channel="graph",
                access_scope=request.access_scope,
                query_options=(
                    {
                        "top_k": max_chunks,
                        "file_scope": file_scope.as_dict(),
                    }
                    if max_chunks and file_scope.enabled
                    else {"top_k": max_chunks}
                    if max_chunks
                    else {"file_scope": file_scope.as_dict()}
                    if file_scope.enabled
                    else None
                ),
            )
        chunks = self._normalize_chunks(payload.get("chunks"))
        if file_scope.enabled:
            chunks = [
                chunk
                for chunk in chunks
                if chunk_in_scope(
                    chunk_metadata=(chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}),
                    scope=file_scope,
                )
            ]
        selected: List[Dict[str, Any]] = []
        for chunk in chunks:
            if max_chunks and len(selected) >= max_chunks:
                break
            selected.append(chunk)
        evidences = self._to_evidences(selected, adapter.metadata().adapter_name)
        if not evidences:
            return ToolResult(
                summary="Chunk scan completed but no high-signal chunks surfaced.",
                diagnostics={"query": query},
            )
        summary = f"Chunk scan surfaced {len(evidences)} candidates for downstream reasoning."
        diagnostics = {
            "query": query,
            "available_chunks": len(chunks),
            "top_k": max_chunks,
        }
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("ChunkScanTool requires a GraphDeepSearchAdapter instance")
        return adapter

    @staticmethod
    def _resolve_query(request: ToolRunRequest) -> str:
        max_chars = int(CHUNK_SCAN_DEFAULT_QUERY_MAX_CHARS)
        if isinstance(request.extra.get("focus_query"), str):
            return clean_query(request.extra["focus_query"], max_chars=max_chars)
        return clean_query(request.question, max_chars=max_chars)

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
