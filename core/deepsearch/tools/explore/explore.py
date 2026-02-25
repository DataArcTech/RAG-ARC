"""Explore tool (graph-first, chunk-aware orchestration)."""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk

from config.core.deepsearch.tool_defaults import EXPLORE_MAX_CONCURRENCY
from config.core.deepsearch import tool_defaults

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER
from .graph_ops import GraphOpsTool
from .locate import LocateTool
from .toc_tree import TocTreeTool
from .tree_tools import TreeRootTool, TreeChildrenTool, TreeNodeTool, TreeOpenTool
from .read_structured import ReadPagesTool
from .web_search import WebSearchTool
from .beam_search import BeamSearchTool
from .llm_chain_explorer import LLMChainExplorerTool
from core.deepsearch.tooling.file_scope_policy import (
    is_global_action_tool,
    strip_file_scope_from_graph_context,
)


_ALLOWED_TOOL_NAMES = {
    "locate",
    "toc.tree",
    "tree.root",
    "tree.children",
    "tree.node",
    "tree.open",
    "graph.ops",
    "graph.beam_search",
    "graph.llm_chain_explorer",
    "web.search",
    "read.pages",
}

_LLM_REQUIRED_ACTIONS = {
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
    """Orchestrate graph-first exploration actions (graph.ops + locate + structured reads)."""

    descriptor = ToolDescriptor(
        name="explore",
        channel="graph",
        description=(
            "Graph-first exploration orchestrator. Runs action lists in parallel: "
            "graph.ops (safe Cypher + templates), locate, web.search, and structured reads "
            "(read.pages). "
            "Good: actions with graph.ops + read.pages. Bad: empty action list."
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
                            "tool": {"type": "string", "description": "Tool name (locate/toc/tree/graph.ops/read.pages)."},
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
                    {
                        "id": "s1",
                        "tool": "locate",
                        "args": {"focus_query": "Company A control Company C", "top_k": 6},
                    },
                    {
                        "id": "g1",
                        "tool": "graph.ops",
                        "args": {"mode": "template", "template": "path_exists", "template_args": {"source": "Company A", "target": "Company C", "max_hops": 4}},
                    },
                    {
                        "id": "r1",
                        "tool": "read.pages",
                        "args": {"file_id": "REPLACE_WITH_REAL_FILE_ID_UUID", "page_start": 2, "page_end": 3, "goal": "verify ownership"},
                    },
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
        tool_overrides: Optional[Dict[str, GraphTool]] = None,
    ):
        self.llm_connector = llm_connector
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.max_concurrency = max(1, int(max_concurrency))
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

        actions, routing_diag = self._enforce_file_routing(actions, graph_context=request.graph_context)
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
        if routing_diag:
            diagnostics["file_routing_gate"] = routing_diag
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
        summary = self._build_summary(results, summaries, routing_diag=routing_diag)
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)

    def _register_builtin_tools(self) -> None:
        if "locate" not in self._tools:
            self._tools["locate"] = LocateTool(
                llm_connector=self.llm_connector,
                dense_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
            )
        if "toc.tree" not in self._tools:
            self._tools["toc.tree"] = TocTreeTool()
        if "tree.root" not in self._tools:
            self._tools["tree.root"] = TreeRootTool()
        if "tree.children" not in self._tools:
            self._tools["tree.children"] = TreeChildrenTool()
        if "tree.node" not in self._tools:
            self._tools["tree.node"] = TreeNodeTool()
        if "tree.open" not in self._tools:
            self._tools["tree.open"] = TreeOpenTool()
        if "graph.ops" not in self._tools:
            self._tools["graph.ops"] = GraphOpsTool()
        if "graph.beam_search" not in self._tools and self.llm_connector is not None:
            self._tools["graph.beam_search"] = BeamSearchTool(llm_connector=self.llm_connector)
        if "graph.llm_chain_explorer" not in self._tools and self.llm_connector is not None:
            self._tools["graph.llm_chain_explorer"] = LLMChainExplorerTool(llm_connector=self.llm_connector)
        if "web.search" not in self._tools:
            self._tools["web.search"] = WebSearchTool()
        if "read.pages" not in self._tools:
            self._tools["read.pages"] = ReadPagesTool()

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
        graph_context = request.graph_context
        if is_global_action_tool(tool_name):
            # Global tools must not inherit prior scoping decisions.
            graph_context = strip_file_scope_from_graph_context(graph_context)
        sub_request = ToolRunRequest(
            question=request.question,
            plan_step=request.plan_step,
            context_evidences=request.context_evidences,
            adapter=request.adapter,
            access_scope=request.access_scope,
            extra=merged_extra,
            graph_context=graph_context,
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

    def _enforce_file_routing(
        self,
        actions: List[Dict[str, Any]],
        *,
        graph_context: Any,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not actions:
            return actions, {}
        for action in actions:
            tool_name = str(action.get("tool") or "").strip()
            if tool_name and tool_name not in _ALLOWED_TOOL_NAMES:
                return actions, {}

        inherited_file_ids: List[str] = []
        try:
            meta = getattr(graph_context, "metadata", None)
            if isinstance(meta, dict):
                scope = meta.get("file_scope")
                if isinstance(scope, dict):
                    raw = scope.get("file_ids") or scope.get("source_file_ids") or []
                    if isinstance(raw, (list, tuple, set, frozenset)):
                        for item in raw:
                            token = str(item or "").strip()
                            if not token:
                                continue
                            try:
                                inherited_file_ids.append(str(uuid.UUID(token)))
                            except Exception:
                                continue
        except Exception:
            inherited_file_ids = []

        def _requires_file_id(tool_name: str, args: Dict[str, Any]) -> bool:
            """Return True when the action semantically requires a file_id/file_ids.

            DeepSearch default is "route file first". To preserve agentic usability, we exempt
            node-id based tree lookups which are already fully pinned to a specific file tree.
            """

            # Global tools do not require a file_id and are often used to *get* routing hints.
            if is_global_action_tool(tool_name):
                return False
            if tool_name == "tree.node":
                return False
            if tool_name == "tree.children":
                # node_id path does not need file_id; section_id path does.
                return not bool(str(args.get("node_id") or "").strip())
            if tool_name == "tree.open":
                # node_id path does not need file_id; section_id fallback does.
                return not bool(str(args.get("node_id") or "").strip())
            return True

        def _looks_like_placeholder(value: str) -> bool:
            token = (value or "").strip()
            return token.startswith("<") and token.endswith(">")

        def _is_valid_uuid(value: str) -> bool:
            token = (value or "").strip()
            if not token or _looks_like_placeholder(token):
                return False
            try:
                uuid.UUID(token)
                return True
            except Exception:  # noqa: BLE001
                return False

        has_valid_file_id = False
        invalid_file_ids: List[str] = []
        has_locate = False
        file_required_tools: List[str] = []
        file_required_actions: List[Dict[str, Any]] = []
        for action in actions:
            tool_name = str(action.get("tool") or "").strip()
            if tool_name == "locate":
                has_locate = True
            args = action.get("args") if isinstance(action.get("args"), dict) else {}
            if _requires_file_id(tool_name, args):
                file_required_actions.append(action)
                file_required_tools.append(tool_name)
            if args:
                file_id = str(args.get("file_id") or "").strip()
                file_ids = args.get("file_ids") or []
                if file_id:
                    if _is_valid_uuid(file_id):
                        has_valid_file_id = True
                    else:
                        invalid_file_ids.append(file_id)
                elif isinstance(file_ids, (list, tuple, set)):
                    for fid in file_ids:
                        token = str(fid or "").strip()
                        if not token:
                            continue
                        if _is_valid_uuid(token):
                            has_valid_file_id = True
                        else:
                            invalid_file_ids.append(token)

        if inherited_file_ids:
            has_valid_file_id = True

        # If the user/model only requested tools that do not require a file_id (currently: node-id tree inspection),
        # do not force locate.
        if not file_required_actions:
            return actions, {}
        if has_valid_file_id:
            return actions, {}

        # When the model passes placeholders / non-UUID file identifiers:
        # - If we already have an inherited file_scope, prefer that scope and strip the invalid ids.
        # - Otherwise, treat as missing and enforce a locate-only round.
        if invalid_file_ids:
            if inherited_file_ids:
                for action in actions:
                    args = action.get("args") if isinstance(action.get("args"), dict) else {}
                    if not args:
                        continue
                    file_id = str(args.get("file_id") or "").strip()
                    if file_id and file_id in invalid_file_ids:
                        args.pop("file_id", None)
                    file_ids = args.get("file_ids")
                    if isinstance(file_ids, list):
                        kept = []
                        for fid in file_ids:
                            token = str(fid or "").strip()
                            if token and token not in invalid_file_ids:
                                kept.append(fid)
                        if kept:
                            args["file_ids"] = kept
                        else:
                            args.pop("file_ids", None)
                    action["args"] = args
                return actions, {
                    "enforced": False,
                    "reason": "invalid_file_id_sanitized",
                    "invalid_file_ids": sorted(set(invalid_file_ids))[:8],
                    "inherited_file_ids": sorted(set(inherited_file_ids))[:8],
                }

            diag = {
                "enforced": True,
                "reason": "invalid_file_id",
                "invalid_file_ids": sorted(set(invalid_file_ids))[:8],
            }
            locate_args = {"top_k": int(tool_defaults.FILE_SEARCH_DEFAULT_TOP_K)}
            return [{"id": "auto_locate", "tool": "locate", "args": locate_args}], diag

        if not has_locate:
            blocked_tools = [str(a.get("tool") or "").strip() for a in file_required_actions if str(a.get("tool") or "").strip()]
            locate_args = {"top_k": int(tool_defaults.FILE_SEARCH_DEFAULT_TOP_K)}
            actions = [{"id": "auto_locate", "tool": "locate", "args": locate_args}]
            return actions, {
                "enforced": True,
                "reason": "missing_file_id",
                "blocked_actions": blocked_tools,
            }

        # locate exists but no file_id in any action: enforce locate only (first step).
        blocked_tools = [
            str(a.get("tool") or "").strip()
            for a in file_required_actions
            if str(a.get("tool") or "").strip() != "locate"
        ]
        locate_actions = [a for a in actions if str(a.get("tool") or "").strip() == "locate"]
        if not locate_actions:
            return actions, {}
        for idx, action in enumerate(locate_actions):
            action["id"] = action.get("id") or f"auto_locate_{idx}"
        return locate_actions, {
            "enforced": True,
            "reason": "file_routing_first",
            "blocked_actions": blocked_tools,
        }

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

    def _requires_llm(self, actions: List[Dict[str, Any]]) -> bool:
        for action in actions:
            tool_name = str(action.get("tool") or "").strip()
            if tool_name in _LLM_REQUIRED_ACTIONS:
                return True
        return False

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
    def _suggest_next_steps_for_action(
        *,
        tool_name: str,
        status: str,
        diagnostics: Dict[str, Any] | None,
    ) -> List[str]:
        """Return short actionable next steps (LLM-visible guidance; generic and domain-agnostic)."""

        diag = diagnostics if isinstance(diagnostics, dict) else {}
        steps: List[str] = []

        if status != "ok":
            reason = str(diag.get("reason") or diag.get("error") or "").strip()
            if reason:
                steps.append(f"Fix the failed {tool_name} call ({reason}), then retry with corrected args.")
            else:
                steps.append(f"Retry {tool_name} with corrected args.")

        if tool_name == "locate":
            steps.extend(
                [
                    "Pick the top file_id (UUID) matching the user's intent; treat others as distractors unless a comparison is requested.",
                    "Navigate within the chosen file via toc.tree or tree.root/tree.open.",
                    "Then call read.pages on likely pages (citeable evidence).",
                ]
            )
        elif tool_name in {"toc.tree", "tree.root", "tree.children", "tree.node", "tree.open"}:
            steps.extend(
                [
                    "Use node_id/section_id/path hints to narrow to the most relevant subtree.",
                    "Use locate(file=<file_id>) to get page-level candidates.",
                    "Then call read.pages for citeable evidence.",
                ]
            )
        elif tool_name.startswith("graph."):
            steps.extend(
                [
                    "Use graph results as derived guidance (entities/relations) to decide where to read in the document.",
                    "Locate supporting pages via toc.tree/tree.* (and locate if needed), then read.pages to ground the claim in source text.",
                ]
            )
        elif tool_name == "read.pages":
            if bool(diag.get("truncated")):
                steps.append("If truncated, narrow the page range or increase read.pages max_chars/max_chunks.")
            steps.append("Use this page evidence for citations; do not cite routing snippets.")
        elif tool_name == "web.search":
            steps.append("If web evidence is used, triangulate with file evidence via read.pages where possible.")

        # Keep it short.
        compacted: List[str] = []
        seen: set[str] = set()
        for s in steps:
            token = str(s or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            compacted.append(token)
            if len(compacted) >= 4:
                break
        return compacted

    @staticmethod
    def _build_summary(
        results: List[_ActionResult],
        summaries: List[str],
        *,
        routing_diag: Dict[str, Any] | None = None,
    ) -> str:
        """Return a model-visible + dev-visible summary.

        This summary is intentionally structured so that:
        - the LLM can reliably read the routing decision + next steps from `summary` (not just diagnostics),
        - developers can audit tool behavior from the same field in traces.
        """

        ok = sum(1 for res in results if res.status == "ok")
        total = len(results)

        gate = routing_diag if isinstance(routing_diag, dict) and routing_diag else None
        if gate:
            reason = str(gate.get("reason") or "").strip()
            if reason == "invalid_file_id":
                next_steps = (
                    "The provided file_id was invalid (placeholder/non-UUID). "
                    "Next: use locate results to pick a real file_id (UUID), then rerun explore with "
                    "toc.tree/tree.root and finally read.pages for citeable evidence."
                )
            elif reason == "invalid_file_id_sanitized":
                next_steps = (
                    "The action args included an invalid file_id, but an inherited file_scope exists. "
                    "Next: rely on the locked file_scope or re-run locate to obtain a real file_id (UUID), "
                    "then proceed with toc.tree/tree.root/tree.open + read.pages."
                )
            elif reason == "file_routing_first":
                next_steps = (
                    "File routing is required before running file-scoped tools. "
                    "Next: pick a file_id from locate results (UUID), then rerun explore with "
                    "toc.tree/tree.root/tree.open + read.pages."
                )
            else:
                next_steps = "Follow the routing_gate guidance, then rerun explore with a real file_id (UUID) and read.pages."
            thinking = f"Routing gate enforced ({reason}). {next_steps}"
            overall_next_steps = [
                "Run explore + locate to obtain a real file_id (UUID).",
                "Navigate to likely pages via toc.tree/tree.root/tree.open, then call read.pages.",
            ]
        else:
            thinking = f"Executed {ok}/{total} explore actions."
            overall_next_steps = []

        action_rows = []
        for res in results:
            row = {
                "id": res.action_id,
                "tool": res.tool,
                "status": res.status,
                "summary": res.summary,
                "error": res.error,
                "evidence_ids": [e.chunk_id for e in (res.evidences or [])],
                "next_steps": ExploreTool._suggest_next_steps_for_action(
                    tool_name=res.tool,
                    status=res.status,
                    diagnostics=res.diagnostics,
                ),
            }
            action_rows.append(row)

        if not overall_next_steps:
            has_pages = any((res.tool == "read.pages" and res.status == "ok") for res in results)
            if has_pages:
                overall_next_steps = [
                    "Proceed to answer/report using read.pages evidence for citations.",
                    "If coverage is insufficient, call read.pages for nearby pages or a different section.",
                ]
            else:
                # If we did not read any pages yet, surface actionable guidance at the top level
                # Individual actions already include next_steps, but models tend to follow the overall list first.
                merged: List[str] = []
                seen: set[str] = set()
                for row in action_rows:
                    for step in row.get("next_steps") or []:
                        token = str(step or "").strip()
                        if not token or token in seen:
                            continue
                        seen.add(token)
                        merged.append(token)
                        if len(merged) >= 4:
                            break
                    if len(merged) >= 4:
                        break
                if merged:
                    overall_next_steps = merged

        payload = {
            "thinking": thinking,
            "answer": {
                "ok_actions": ok,
                "total_actions": total,
                "routing_gate": gate,
                "next_steps": overall_next_steps,
                # Tool discoverability: list all allowed Explore actions so models
                # (and developers reading traces) know what can be scheduled next without relying
                # on hidden tool catalogs. This is intentionally a small stable set.
                "allowed_action_tools": sorted(_ALLOWED_TOOL_NAMES),
                "actions": action_rows,
                "notes": "Action summaries are navigation unless source=read.pages (citeable).",
            },
        }
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            # Fallback: keep old behavior if serialization fails.
            if summaries:
                return f"explore completed {ok}/{total} actions. " + " ".join(summaries)
            return f"explore completed {ok}/{total} actions."
