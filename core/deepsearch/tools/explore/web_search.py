"""Web search tool (Tavily-backed)."""
import os
from typing import Any, Dict, List

from config.core.deepsearch import tool_defaults
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_WEB_SNIPPET
from encapsulation.data_model.deepsearch import EvidenceChunk
from encapsulation.web_search import TavilySearchClient, TavilySearchResult
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.deepsearch.utils.evidence_ids import hashed_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_DERIVED, SCOPE_OWNER


class WebSearchTool(GraphTool):
    """Local web.search tool backed by Tavily."""

    descriptor = ToolDescriptor(
        name="web.search",
        channel="web",
        description=(
            "Tavily-backed web search for realtime/current info only. "
            "Use after graph-first tools when internal evidence is insufficient."
        ),
        speed="medium",
        cost="medium",
        strategy_tags=("web", "search", EVIDENCE_DERIVED, SCOPE_OWNER),
        profile="X",
        determinism="stochastic",
        namespace="rag-arc.deepsearch.tools.explore.web.search",
        mcp_callable=False,
        input_schema=build_input_schema(
            extra_properties={
                "query": {"type": "string", "description": "Search query (required)."},
                "max_results": {"type": "integer", "minimum": 1, "description": "Override max results."},
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "Tavily search depth.",
                },
            },
            required_extra_fields=("query",),
        ),
        example_args={
            "question": "今天美元兑人民币汇率是多少？",
            "plan_step": "plan_web_01",
            "extra": {"query": "USD CNY exchange rate today", "max_results": 5},
        },
    )

    def __init__(
        self,
        *,
        client: TavilySearchClient | None = None,
        api_key: str | None = None,
        endpoint_url: str | None = None,
        max_results: int = tool_defaults.WEB_SEARCH_DEFAULT_MAX_RESULTS,
        timeout_seconds: float = tool_defaults.WEB_SEARCH_DEFAULT_TIMEOUT_SECONDS,
        search_depth: str = tool_defaults.WEB_SEARCH_DEFAULT_SEARCH_DEPTH,
    ) -> None:
        self.max_results = int(max_results)
        self.timeout_seconds = float(timeout_seconds)
        self.search_depth = str(search_depth or tool_defaults.WEB_SEARCH_DEFAULT_SEARCH_DEPTH).strip().lower()
        self._client_override = client is not None
        self.api_key = str(api_key or os.getenv("TAVILY_API_KEY") or "").strip() or None
        self.endpoint_url = str(endpoint_url or os.getenv("TAVILY_ENDPOINT_URL") or "").strip() or None
        self.client = client or TavilySearchClient(
            api_key=self.api_key,
            endpoint_url=self.endpoint_url or "https://api.tavily.com/search",
            timeout_seconds=self.timeout_seconds,
            search_depth="advanced" if self.search_depth != "basic" else "basic",
            max_results=self.max_results,
        )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        extra = request.extra or {}
        query = str(extra.get("query") or "").strip()
        if not query:
            return ToolResult(
                summary="web.search skipped: missing query.",
                diagnostics={"reason": "missing_query"},
            )

        max_results = self._resolve_max_results(extra.get("max_results"))
        search_depth = str(extra.get("search_depth") or self.search_depth).strip().lower()
        if search_depth not in {"basic", "advanced"}:
            search_depth = self.search_depth

        results: List[TavilySearchResult]
        try:
            client = self.client
            if not self._client_override and search_depth != self.search_depth:
                client = TavilySearchClient(
                    api_key=self.api_key,
                    endpoint_url=self.endpoint_url or "https://api.tavily.com/search",
                    timeout_seconds=self.timeout_seconds,
                    search_depth=search_depth,  # type: ignore[arg-type]
                    max_results=max_results,
                )
            results = await client.asearch(query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                summary=f"web.search failed: {exc}",
                diagnostics={"reason": "provider_error", "error": str(exc), "query": query},
            )

        evidences: List[EvidenceChunk] = []
        results_payload: List[Dict[str, Any]] = []
        max_chars = int(tool_defaults.WEB_SEARCH_SNIPPET_MAX_CHARS)
        for idx, item in enumerate(results):
            content = f"{item.title}\n{item.content}".strip()
            if max_chars > 0 and len(content) > max_chars:
                content = content[: max_chars - 3].rstrip() + "..."
            chunk_id = hashed_chunk_id(source="web.search", content=f"{query}::{idx}::{item.url or ''}")
            evidences.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source="web.search",
                    content=content,
                    kind=EVIDENCE_KIND_DERIVED,
                    score=item.score,
                    provenance={
                        "provider": "tavily",
                        "url": item.url,
                        "query": query,
                        "rank": idx,
                        "evidence_class": EVIDENCE_CLASS_WEB_SNIPPET,
                    },
                )
            )
            results_payload.append(
                {
                    "title": item.title,
                    "url": item.url,
                    "score": item.score,
                    "rank": idx,
                }
            )

        summary = f"web.search returned {len(evidences)} results." if evidences else "web.search returned no results."
        diagnostics = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "results": results_payload,
        }
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)

    @staticmethod
    def _resolve_max_results(raw: Any) -> int:
        try:
            value = int(raw) if raw is not None else int(tool_defaults.WEB_SEARCH_DEFAULT_MAX_RESULTS)
        except (TypeError, ValueError):
            value = int(tool_defaults.WEB_SEARCH_DEFAULT_MAX_RESULTS)
        value = max(1, value)
        return min(value, int(tool_defaults.WEB_SEARCH_MAX_RESULTS))
