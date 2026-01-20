"""Unified graph.ops tool (safe Cypher + template library)."""
import re
from typing import Any, Dict, Optional
from config.core.deepsearch.tool_defaults import (
    GRAPH_OPS_ALLOW_CUSTOM_CYPHER,
    GRAPH_OPS_MAX_CYPHER_CHARS,
    GRAPH_OPS_MAX_ROWS,
    GRAPH_OPS_REQUIRE_OWNER_FILTER,
)
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.graph_adapter.concurrency import adapter_locked
from core.graph_adapter.cypher import adapter_supports_cypher, assert_read_only_cypher

from ...base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ...governance_tags import EVIDENCE_PRIMARY, REQUIRES_CYPHER, SCOPE_OWNER
from .templates import template_descriptions, template_names, template_registry
from .templates_utils import build_derived_evidence


_OWNER_TOKEN_RE = re.compile(r"\$owner_id|\$global_owner")


class GraphOpsTool(GraphTool):
    """Safe Cypher runner with a template library for deterministic graph reasoning."""

    def __init__(
        self,
        *,
        allow_custom_cypher: bool = GRAPH_OPS_ALLOW_CUSTOM_CYPHER,
        require_owner_filter: bool = GRAPH_OPS_REQUIRE_OWNER_FILTER,
        max_cypher_chars: int = GRAPH_OPS_MAX_CYPHER_CHARS,
        max_rows: int = GRAPH_OPS_MAX_ROWS,
        templates: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.allow_custom_cypher = bool(allow_custom_cypher)
        self.require_owner_filter = bool(require_owner_filter)
        self.max_cypher_chars = max(256, int(max_cypher_chars))
        self.max_rows = max(1, int(max_rows))
        self._templates = templates or template_registry()

    descriptor = ToolDescriptor(
        name="graph.ops",
        channel="graph",
        description=(
            "Unified graph operator: run read-only Cypher safely or use a named template "
            "for deterministic reasoning (paths, intersections, aggregates, rule checks, "
            "relation-path explore/ground, schema/terms). "
            "Good: template=path_exists with source/target. "
            "Bad: custom cypher without $owner_id/$global_owner or any write."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("graph_ops", "cypher", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_ops",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "mode": {
                    "type": "string",
                    "enum": ["template", "cypher"],
                    "description": "Execution mode: template (recommended) or custom read-only cypher.",
                },
                "template": {
                    "type": "string",
                    "description": "Template name (see template catalog).",
                    "enum": template_names(),
                },
                "template_args": {"type": "object", "description": "Template-specific arguments."},
                "cypher": {"type": "string", "description": "Custom read-only Cypher query."},
                "params": {"type": "object", "description": "Parameters for custom Cypher."},
                "max_rows": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Override max rows returned (applies to custom cypher diagnostics).",
                },
            }
        ),
        example_args={
            "question": "Find 2-hop ownership path between A and C",
            "plan_step": "plan_01",
            "extra": {"mode": "template", "template": "path_exists", "template_args": {"source": "A公司", "target": "C公司", "max_hops": 4}},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = request.adapter
        if adapter is None or not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="graph.ops requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )
        if request.access_scope is None:
            return ToolResult(summary="graph.ops requires an access scope (owner).")

        extra = request.extra or {}
        mode = str(extra.get("mode") or "").strip().lower()
        if not mode:
            mode = "cypher" if extra.get("cypher") else "template"

        if mode == "template":
            template_name = str(extra.get("template") or "").strip()
            if not template_name:
                return ToolResult(
                    summary="graph.ops requires template name when mode=template.",
                    diagnostics={"available_templates": template_names()},
                )
            template = self._templates.get(template_name)
            if template is None:
                return ToolResult(
                    summary=f"graph.ops template '{template_name}' is not registered.",
                    diagnostics={"available_templates": template_names()},
                )
            args = self._extract_template_args(extra)
            return await template.handler(self, request, args)

        if mode == "cypher":
            if not self.allow_custom_cypher:
                return ToolResult(summary="Custom Cypher is disabled for graph.ops.")
            cypher = str(extra.get("cypher") or "").strip()
            if not cypher:
                return ToolResult(summary="graph.ops requires a non-empty cypher string when mode=cypher.")
            params = extra.get("params") if isinstance(extra.get("params"), dict) else {}
            max_rows = self._resolve_max_rows(extra.get("max_rows"))
            return await self._run_custom_cypher(request, cypher, params, max_rows=max_rows)

        return ToolResult(summary="graph.ops mode must be 'template' or 'cypher'.")

    def _extract_template_args(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(extra.get("template_args"), dict):
            return dict(extra["template_args"])
        ignored = {"mode", "template", "cypher", "params", "max_rows"}
        return {key: value for key, value in extra.items() if key not in ignored}

    def _resolve_max_rows(self, raw: Any) -> int:
        try:
            value = int(raw) if raw is not None else self.max_rows
        except (TypeError, ValueError):
            value = self.max_rows
        return max(1, min(self.max_rows, value))

    async def _run_custom_cypher(
        self,
        request: ToolRunRequest,
        cypher: str,
        params: Dict[str, Any],
        *,
        max_rows: int,
    ) -> ToolResult:
        if len(cypher) > self.max_cypher_chars:
            return ToolResult(summary="Custom Cypher exceeds maximum length.")
        if self.require_owner_filter and not _OWNER_TOKEN_RE.search(cypher):
            return ToolResult(
                summary="Custom Cypher must reference $owner_id and $global_owner for safety.",
                diagnostics={"reason": "owner_filter_missing"},
            )
        if {"owner_id", "global_owner"} & set(params.keys()):
            return ToolResult(
                summary="Custom Cypher params cannot override reserved owner scope keys.",
                diagnostics={"reason": "reserved_param"},
            )

        rows = await self._acypher(request, cypher, params)
        preview = rows[:max_rows] if isinstance(rows, list) else []
        summary = f"cypher rows={len(rows or [])}"
        content = summary
        evidence = build_derived_evidence(
            tool_name=self.descriptor.name,
            plan_step=request.plan_step,
            label="cypher",
            content=content,
            provenance={"cypher": cypher, "rows_preview": preview},
            kind=EVIDENCE_KIND_DERIVED,
        )
        return ToolResult(
            summary=summary,
            evidences=[evidence],
            diagnostics={"rows": preview, "row_count": len(rows or []), "mode": "cypher"},
        )

    async def _acypher(self, request: ToolRunRequest, cypher: str, params: Dict[str, Any]) -> list[Dict[str, Any]]:
        if len(cypher) > self.max_cypher_chars:
            raise RuntimeError("Cypher exceeds maximum length for graph.ops.")
        assert_read_only_cypher(str(cypher or ""))
        if self.require_owner_filter and not _OWNER_TOKEN_RE.search(cypher):
            raise RuntimeError("Cypher query missing owner_id/global_owner filter tokens.")
        adapter = request.adapter
        if adapter is None:
            raise RuntimeError("graph.ops requires a GraphDeepSearchAdapter instance")

        async with adapter_locked(adapter):
            return await adapter.acypher(cypher, params, access_scope=request.access_scope)

    def template_catalog(self) -> Dict[str, str]:
        return template_descriptions()
