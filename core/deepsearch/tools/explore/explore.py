"""Explore tool (graph-first, chunk-aware orchestration)."""
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk

from config.core.deepsearch.tool_defaults import EXPLORE_MAX_CONCURRENCY, EXPLORE_READ_MAX_CHARS, EXPLORE_READ_MAX_CHUNKS
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_SOURCE_TEXT
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_PRIMARY
from core.graph_adapter.cypher import adapter_supports_cypher
from core.graph_adapter.concurrency import adapter_locked

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER
from .graph_ops import GraphOpsTool
from .search.file_search import FileSearchTool
from .search.section_search import SectionSearchTool
from .search.tool import SearchGlobalTool, SearchScopedTool
from .search.faiss import SearchFaissTool, SearchGlobalFaissTool
from .search.bm25 import SearchBM25Tool, SearchGlobalBM25Tool
from .search.graph_chunk import SearchGraphChunkTool, SearchGlobalGraphTool
from .toc_tree import TocTreeTool
from .read_structured import ReadPagesTool, ReadSectionTool
from .read_neighbors import ReadNeighborsTool
from .web_search import WebSearchTool
from .beam_search import BeamSearchTool
from .llm_chain_explorer import LLMChainExplorerTool


_ALLOWED_TOOL_NAMES = {
    "search.file",
    "search.section",
    "toc.tree",
    "search.scoped",
    "search.global",
    "search.scoped.faiss",
    "search.scoped.bm25",
    "search.scoped.graph",
    "search.global.faiss",
    "search.global.bm25",
    "search.global.graph",
    "graph.ops",
    "graph.beam_search",
    "graph.llm_chain_explorer",
    "web.search",
    "read.chunk",
    "read.section",
    "read.pages",
    "read.neighbors",
}

_LLM_REQUIRED_ACTIONS = {
    "search.scoped.graph",
    "search.global.graph",
    "graph.beam_search",
    "graph.llm_chain_explorer",
}


@dataclass
class _ActionResult:
    action_id: str
    tool: str
    status: str
    summary: str
    evidences: List[EvidenceChunk]
    diagnostics: Dict[str, Any]
    error: Optional[str] = None


class ExploreTool(GraphTool):
    """Orchestrate graph-first exploration actions (graph.ops + search + read.chunk)."""

    descriptor = ToolDescriptor(
        name="explore",
        channel="graph",
        description=(
            "Graph-first exploration orchestrator. Runs action lists in parallel: "
            "graph.ops (safe Cypher + templates), search, web.search, and read.chunk "
            "(full chunk read by chunk_id + goal). "
            "Good: actions with graph.ops + read.chunk. Bad: empty action list."
        ),
        speed="medium",
        cost="medium",
        strategy_tags=("explore", "graph_first", EVIDENCE_PRIMARY, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="X",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.explore",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Optional action id for tracking."},
                            "tool": {"type": "string", "description": "Tool name (search/graph.ops/read.chunk)."},
                            "args": {"type": "object", "description": "Tool-specific args (passed as extra)."},
                        },
                        "required": ["tool"],
                    },
                    "description": "Action list executed in parallel.",
                },
                "max_concurrency": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Override max parallel actions (capped by safe defaults).",
                },
            }
        ),
        example_args={
            "question": "Does Company A indirectly control Company C?",
            "plan_step": "plan_01",
            "extra": {
                "actions": [
                    {"id": "s1", "tool": "search.scoped", "args": {"focus_query": "Company A control Company C", "file_id": "<file_id>", "top_k": 6}},
                    {
                        "id": "g1",
                        "tool": "graph.ops",
                        "args": {"mode": "template", "template": "path_exists", "template_args": {"source": "Company A", "target": "Company C", "max_hops": 4}},
                    },
                    {"id": "r1", "tool": "read.chunk", "args": {"chunk_ids": ["c1"], "goal": "verify ownership"}},
                ]
            },
        },
    )

    def __init__(
        self,
        llm_connector=None,
        *,
        dense_retriever=None,
        bm25_retriever=None,
        max_concurrency: int = EXPLORE_MAX_CONCURRENCY,
        read_max_chunks: int = EXPLORE_READ_MAX_CHUNKS,
        read_max_chars: int = EXPLORE_READ_MAX_CHARS,
        tool_overrides: Optional[Dict[str, GraphTool]] = None,
    ):
        self.llm_connector = llm_connector
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.max_concurrency = max(1, int(max_concurrency))
        self.read_max_chunks = max(1, int(read_max_chunks))
        self.read_max_chars = max(120, int(read_max_chars))
        self._tools: Dict[str, GraphTool] = {}
        if tool_overrides:
            self._tools.update(tool_overrides)
        self._register_builtin_tools()

    async def run(self, request: ToolRunRequest) -> ToolResult:
        actions = self._normalize_actions(request.extra.get("actions"))
        if not actions:
            return ToolResult(
                summary="explore skipped: no actions provided.",
                diagnostics={"actions": [], "errors": ["missing_actions"]},
            )

        if self.llm_connector is None and self._requires_llm(actions):
            raise RuntimeError("explore requires an LLM connector for requested actions.")

        max_concurrency = self._resolve_concurrency(request.extra.get("max_concurrency"))
        semaphore = asyncio.Semaphore(max_concurrency)

        tasks = [
            asyncio.create_task(self._run_action(action=action, request=request, semaphore=semaphore))
            for action in actions
        ]
        results = await asyncio.gather(*tasks)

        evidences: List[EvidenceChunk] = []
        summaries: List[str] = []
        diagnostics: Dict[str, Any] = {"actions": []}
        errors: List[str] = []
        for result in results:
            diagnostics["actions"].append(self._action_diagnostics(result))
            if result.evidences:
                evidences.extend(result.evidences)
            if result.summary:
                summaries.append(result.summary)
            if result.error:
                errors.append(result.error)

        diagnostics["errors"] = errors
        summary = self._build_summary(results, summaries)
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)

    def _register_builtin_tools(self) -> None:
        if "search.file" not in self._tools:
            self._tools["search.file"] = FileSearchTool()
        if "search.section" not in self._tools:
            self._tools["search.section"] = SectionSearchTool()
        if "toc.tree" not in self._tools:
            self._tools["toc.tree"] = TocTreeTool()
        if "search.scoped" not in self._tools:
            self._tools["search.scoped"] = SearchScopedTool(
                llm_connector=self.llm_connector,
                dense_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
            )
        if "search.global" not in self._tools:
            self._tools["search.global"] = SearchGlobalTool(
                llm_connector=self.llm_connector,
                dense_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
            )
        if "search.scoped.faiss" not in self._tools:
            self._tools["search.scoped.faiss"] = SearchFaissTool(dense_retriever=self.dense_retriever)
        if "search.scoped.bm25" not in self._tools:
            self._tools["search.scoped.bm25"] = SearchBM25Tool(bm25_retriever=self.bm25_retriever)
        if "search.scoped.graph" not in self._tools:
            self._tools["search.scoped.graph"] = SearchGraphChunkTool(llm_connector=self.llm_connector)
        if "search.global.faiss" not in self._tools:
            self._tools["search.global.faiss"] = SearchGlobalFaissTool(dense_retriever=self.dense_retriever)
        if "search.global.bm25" not in self._tools:
            self._tools["search.global.bm25"] = SearchGlobalBM25Tool(bm25_retriever=self.bm25_retriever)
        if "search.global.graph" not in self._tools:
            self._tools["search.global.graph"] = SearchGlobalGraphTool(llm_connector=self.llm_connector)
        if "graph.ops" not in self._tools:
            self._tools["graph.ops"] = GraphOpsTool()
        if "graph.beam_search" not in self._tools and self.llm_connector is not None:
            self._tools["graph.beam_search"] = BeamSearchTool(llm_connector=self.llm_connector)
        if "graph.llm_chain_explorer" not in self._tools and self.llm_connector is not None:
            self._tools["graph.llm_chain_explorer"] = LLMChainExplorerTool(llm_connector=self.llm_connector)
        if "web.search" not in self._tools:
            self._tools["web.search"] = WebSearchTool()
        if "read.section" not in self._tools:
            self._tools["read.section"] = ReadSectionTool()
        if "read.pages" not in self._tools:
            self._tools["read.pages"] = ReadPagesTool()
        if "read.neighbors" not in self._tools:
            self._tools["read.neighbors"] = ReadNeighborsTool()

    async def _run_action(
        self,
        *,
        action: Dict[str, Any],
        request: ToolRunRequest,
        semaphore: asyncio.Semaphore,
    ) -> _ActionResult:
        action_id = str(action.get("id") or "").strip() or f"action_{action.get('index')}"
        tool_name = str(action.get("tool") or "").strip()
        args = action.get("args") if isinstance(action.get("args"), dict) else {}

        if tool_name not in _ALLOWED_TOOL_NAMES:
            return _ActionResult(
                action_id=action_id,
                tool=tool_name,
                status="failed",
                summary=f"{tool_name}: rejected (tool not allowed).",
                evidences=[],
                diagnostics={"reason": "tool_not_allowed"},
                error=f"{tool_name}: tool_not_allowed",
            )

        async with semaphore:
            start = time.perf_counter()
            if tool_name == "read.chunk":
                result = await self._read_chunk(action_id=action_id, args=args, request=request)
            else:
                result = await self._delegate_tool(action_id=action_id, tool_name=tool_name, args=args, request=request)
            latency_ms = int((time.perf_counter() - start) * 1000)
            result.diagnostics.setdefault("latency_ms", latency_ms)
            return result

    async def _delegate_tool(
        self,
        *,
        action_id: str,
        tool_name: str,
        args: Dict[str, Any],
        request: ToolRunRequest,
    ) -> _ActionResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return _ActionResult(
                action_id=action_id,
                tool=tool_name,
                status="failed",
                summary=f"{tool_name}: tool unavailable.",
                evidences=[],
                diagnostics={"reason": "tool_unavailable"},
                error=f"{tool_name}: tool_unavailable",
            )
        merged_extra = self._merge_extra(request.extra, args)
        sub_request = ToolRunRequest(
            question=request.question,
            plan_step=request.plan_step,
            context_evidences=request.context_evidences,
            adapter=request.adapter,
            access_scope=request.access_scope,
            extra=merged_extra,
            graph_context=request.graph_context,
            coverage_metrics=request.coverage_metrics,
        )
        try:
            tool_result = await tool.run(sub_request)
        except Exception as exc:  # noqa: BLE001
            return _ActionResult(
                action_id=action_id,
                tool=tool_name,
                status="failed",
                summary=f"{tool_name}: execution failed.",
                evidences=[],
                diagnostics={"reason": "exception", "error": str(exc)},
                error=f"{tool_name}: {exc}",
            )
        return _ActionResult(
            action_id=action_id,
            tool=tool_name,
            status="ok",
            summary=tool_result.summary,
            evidences=tool_result.evidences,
            diagnostics=tool_result.diagnostics,
        )

    async def _read_chunk(
        self,
        *,
        action_id: str,
        args: Dict[str, Any],
        request: ToolRunRequest,
    ) -> _ActionResult:
        if request.adapter is None:
            return _ActionResult(
                action_id=action_id,
                tool="read.chunk",
                status="failed",
                summary="read.chunk failed: adapter missing.",
                evidences=[],
                diagnostics={"reason": "adapter_missing"},
                error="read.chunk: adapter_missing",
            )

        if not adapter_supports_cypher(request.adapter):
            return _ActionResult(
                action_id=action_id,
                tool="read.chunk",
                status="failed",
                summary="read.chunk failed: Cypher unavailable.",
                evidences=[],
                diagnostics={"reason": "cypher_unavailable"},
                error="read.chunk: cypher_unavailable",
            )

        chunk_ids = self._normalize_chunk_ids(args.get("chunk_ids"))
        if not chunk_ids:
            return _ActionResult(
                action_id=action_id,
                tool="read.chunk",
                status="skipped",
                summary="read.chunk skipped: no chunk_ids.",
                evidences=[],
                diagnostics={"reason": "empty_chunk_ids"},
            )
        if len(chunk_ids) > self.read_max_chunks:
            chunk_ids = chunk_ids[: self.read_max_chunks]

        goal = str(args.get("goal") or "").strip() or None
        max_chars = self._resolve_read_max_chars(args.get("max_chars"))
        file_scope_ids = self._normalize_file_scope_ids(args)

        cypher = (
            "MATCH (c:Chunk)\n"
            "WHERE c.chunk_id IN $chunk_ids AND COALESCE(c.owner_id, $global_owner) = $owner_id\n"
            "RETURN c.chunk_id AS chunk_id, c.content AS content, c.metadata AS metadata, "
            "c.source_file_id AS source_file_id, c.owner_id AS owner_id\n"
        )
        # Protect Neo4j driver calls when the adapter does not opt into concurrency.
        async with adapter_locked(request.adapter):
            rows = await request.adapter.acypher(cypher, {"chunk_ids": chunk_ids}, access_scope=request.access_scope)
        evidences: List[EvidenceChunk] = []
        seen: set[str] = set()
        dropped_out_of_scope = 0
        dropped_missing_file_id = 0
        dropped_ids: List[str] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            content = str(row.get("content") or "")
            if len(content) > max_chars:
                content = content[:max_chars].rstrip() + "..."
            metadata = self._parse_metadata(row.get("metadata"))
            source_file_id = row.get("source_file_id") or metadata.get("source_file_id")
            if source_file_id:
                metadata["source_file_id"] = source_file_id
            if file_scope_ids:
                if not source_file_id:
                    dropped_missing_file_id += 1
                    dropped_ids.append(chunk_id)
                    continue
                if str(source_file_id) not in file_scope_ids:
                    dropped_out_of_scope += 1
                    dropped_ids.append(chunk_id)
                    continue
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source="explore.read.chunk",
                content=content,
                kind=EVIDENCE_KIND_PRIMARY,
                provenance={
                    "goal": goal,
                    "chunk_id": chunk_id,
                    "evidence_class": EVIDENCE_CLASS_SOURCE_TEXT,
                    "metadata": metadata,
                },
                score=None,
            )
            evidences.append(evidence)

        summary = f"read.chunk returned {len(evidences)} chunks." if evidences else "read.chunk returned no chunks."
        diagnostics = {"requested": len(chunk_ids), "returned": len(evidences), "goal": goal}
        if file_scope_ids:
            diagnostics["file_scope"] = {"file_ids": sorted(file_scope_ids), "source": "tool_args"}
            diagnostics["dropped_out_of_scope"] = dropped_out_of_scope
            diagnostics["dropped_missing_file_id"] = dropped_missing_file_id
            if dropped_ids:
                diagnostics["dropped_chunk_ids"] = dropped_ids[:20]
            if dropped_out_of_scope or dropped_missing_file_id:
                summary = (
                    summary.rstrip(".")
                    + f"; dropped {dropped_out_of_scope} out-of-scope"
                    + (f" (+{dropped_missing_file_id} missing file_id)" if dropped_missing_file_id else "")
                    + "."
                )
        return _ActionResult(
            action_id=action_id,
            tool="read.chunk",
            status="ok",
            summary=summary,
            evidences=evidences,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _normalize_file_scope_ids(args: Any) -> set[str]:
        if not isinstance(args, dict):
            return set()
        out: set[str] = set()
        for key in ("file_ids", "file_id", "source_file_ids", "source_file_id"):
            raw = args.get(key)
            if raw is None:
                continue
            items = raw if isinstance(raw, (list, tuple, set, frozenset)) else [raw]
            for item in items:
                token = str(item or "").strip()
                if token:
                    out.add(token)
        return out

    @staticmethod
    def _normalize_actions(actions: Any) -> List[Dict[str, Any]]:
        if not isinstance(actions, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(actions):
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry["index"] = idx
            normalized.append(entry)
        return normalized

    @staticmethod
    def _normalize_chunk_ids(raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, dict):
            return []
        if not isinstance(raw, Iterable):
            return []
        out: List[str] = []
        for value in raw:
            token = str(value or "").strip()
            if token:
                out.append(token)
        return out

    @staticmethod
    def _merge_extra(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base or {})
        merged.pop("actions", None)
        merged.update(override or {})
        return merged

    def _resolve_concurrency(self, override: Any) -> int:
        try:
            value = int(override) if override is not None else self.max_concurrency
        except (TypeError, ValueError):
            value = self.max_concurrency
        return max(1, min(value, self.max_concurrency))

    def _resolve_read_max_chars(self, override: Any) -> int:
        try:
            value = int(override) if override is not None else self.read_max_chars
        except (TypeError, ValueError):
            value = self.read_max_chars
        return max(120, min(value, self.read_max_chars))

    def _requires_llm(self, actions: List[Dict[str, Any]]) -> bool:
        for action in actions:
            tool_name = str(action.get("tool") or "").strip()
            if tool_name in _LLM_REQUIRED_ACTIONS:
                return True
            if tool_name == "search":
                args = action.get("args") if isinstance(action.get("args"), dict) else {}
                channels = args.get("channels")
                if channels is None:
                    return True
                if isinstance(channels, str):
                    channels = [c.strip() for c in channels.split(",") if c.strip()]
                if isinstance(channels, (list, tuple, set)):
                    if "graph_chunk" in {str(c).strip() for c in channels if str(c).strip()}:
                        return True
        return False

    @staticmethod
    def _parse_metadata(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                return dict(parsed) if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _action_diagnostics(result: _ActionResult) -> Dict[str, Any]:
        payload = {
            "id": result.action_id,
            "tool": result.tool,
            "status": result.status,
            "summary": result.summary,
            "diagnostics": result.diagnostics,
        }
        if result.error:
            payload["error"] = result.error
        return payload

    @staticmethod
    def _build_summary(results: List[_ActionResult], summaries: List[str]) -> str:
        ok = sum(1 for res in results if res.status == "ok")
        total = len(results)
        if summaries:
            return f"explore completed {ok}/{total} actions. " + " ".join(summaries)
        return f"explore completed {ok}/{total} actions."
