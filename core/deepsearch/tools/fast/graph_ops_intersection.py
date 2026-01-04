"""Deterministic intersection query tool backed by Neo4j Cypher."""
from typing import Any, Dict, List, Sequence

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.graph_adapter.cypher import GraphCypherQueryable
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from .graph_ops_common import (
    directionality_config,
    enforce_direction_for_sensitive_predicates,
    enforce_undirected_for_non_sensitive_predicates,
    limit_int,
    normalize_entity_name,
    normalize_predicates,
    rel_pattern,
)


class GraphIntersectionTool(GraphTool):
    """Intersection query (common neighbors) for deterministic multi-hop reasoning."""

    descriptor = ToolDescriptor(
        name="graph.intersection",
        channel="graph",
        description=(
            "Deterministic intersection query (shared neighbors/targets) backed by Neo4j Cypher. "
            "Requires `Entity.entity_name_normalized` for matching and `RELATES_TO` fact edges."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("intersection", "deterministic", "cypher"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_intersection",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "left": {"type": "string", "description": "Left entity name (normalized match)."},
                "left_type": {"type": "string", "description": "Optional left entity_type for disambiguation."},
                "right": {"type": "string", "description": "Right entity name (normalized match)."},
                "right_type": {"type": "string", "description": "Optional right entity_type for disambiguation."},
                "left_predicates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Allowed predicates from left -> target.",
                },
                "right_predicates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Allowed predicates from right -> target.",
                },
                "left_direction": {
                    "type": "string",
                    "description": "Edge direction for the left side (out|in|both). Defaults to `direction`.",
                    "enum": ["out", "in", "both"],
                },
                "right_direction": {
                    "type": "string",
                    "description": "Edge direction for the right side (out|in|both). Defaults to `direction`.",
                    "enum": ["out", "in", "both"],
                },
                "direction": {
                    "type": "string",
                    "description": "Edge direction for matching (out|in|both).",
                    "enum": ["out", "in", "both"],
                },
                "limit": {"type": "integer", "description": "Max intersection nodes returned.", "minimum": 1},
            },
            required_extra_fields=("left", "right"),
        ),
        example_args={
            "question": "Do Zenthorax and Vira-X have a shared target that indicates DDI risk?",
            "plan_step": "plan_07",
            "extra": {
                "left": "Zenthorax",
                "right": "Vira-X",
                "left_predicates": ["INHIBITS"],
                "right_predicates": ["METABOLIZED_BY"],
                "direction": "out",
            },
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not isinstance(adapter, GraphCypherQueryable):
            return ToolResult(summary="Intersection skipped because the adapter does not support Cypher queries.")

        left = normalize_entity_name(request.extra.get("left"))
        right = normalize_entity_name(request.extra.get("right"))
        if not left or not right:
            return ToolResult(summary="Intersection requires non-empty left/right entity names.")

        left_type = str(request.extra.get("left_type") or "").strip()
        right_type = str(request.extra.get("right_type") or "").strip()
        left_preds = normalize_predicates(request.extra.get("left_predicates"))
        right_preds = normalize_predicates(request.extra.get("right_predicates"))
        direction_default = str(request.extra.get("direction") or "out")
        left_direction_raw = str(request.extra.get("left_direction") or direction_default)
        right_direction_raw = str(request.extra.get("right_direction") or direction_default)
        directionality = directionality_config(adapter)
        left_direction, left_forced = enforce_direction_for_sensitive_predicates(
            left_direction_raw, left_preds, directionality=directionality, default_direction="out"
        )
        left_direction, left_forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            left_direction, left_preds, directionality=directionality
        )
        right_direction, right_forced = enforce_direction_for_sensitive_predicates(
            right_direction_raw, right_preds, directionality=directionality, default_direction="out"
        )
        right_direction, right_forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            right_direction, right_preds, directionality=directionality
        )
        limit = limit_int(request.extra.get("limit"), 12, max_value=50)

        rel_left = rel_pattern(left_direction, rel_var="lr", rel_type="RELATES_TO")
        rel_right = rel_pattern(right_direction, rel_var="rr", rel_type="RELATES_TO")
        cypher = f"""
        MATCH (l0:Entity)
        WHERE COALESCE(l0.owner_id, $global_owner) = $owner_id
          AND l0.entity_name_normalized = $left
          AND ($left_type = '' OR l0.entity_type = $left_type)
        WITH collect(l0) AS left_nodes
        WITH size(left_nodes) AS left_candidates,
             CASE WHEN size(left_nodes) = 1 THEN left_nodes[0] ELSE NULL END AS l
        MATCH (r0:Entity)
        WHERE COALESCE(r0.owner_id, $global_owner) = $owner_id
          AND r0.entity_name_normalized = $right
          AND ($right_type = '' OR r0.entity_type = $right_type)
        WITH left_candidates, l, collect(r0) AS right_nodes
        WITH left_candidates,
             size(right_nodes) AS right_candidates,
             l,
             CASE WHEN size(right_nodes) = 1 THEN right_nodes[0] ELSE NULL END AS r
        CALL {{
          WITH left_candidates, right_candidates, l, r
          WITH left_candidates, right_candidates, l, r
          WHERE left_candidates = 1 AND right_candidates = 1
          MATCH (l){rel_left}(t:Entity)
          WHERE COALESCE(lr.owner_id, $global_owner) = $owner_id
            AND COALESCE(t.owner_id, $global_owner) = $owner_id
          MATCH (r){rel_right}(t)
          WHERE COALESCE(rr.owner_id, $global_owner) = $owner_id
            AND COALESCE(t.owner_id, $global_owner) = $owner_id
          {self._predicate_filters(left_alias="lr", right_alias="rr", left_preds=left_preds, right_preds=right_preds)}
          RETURN left_candidates AS left_candidates,
                 right_candidates AS right_candidates,
                 t.entity_name AS target,
                 collect(DISTINCT lr.fact_id) AS left_fact_ids,
                 collect(DISTINCT rr.fact_id) AS right_fact_ids,
                 collect(DISTINCT lr.source_chunk_ids) AS left_source_chunk_ids,
                 collect(DISTINCT rr.source_chunk_ids) AS right_source_chunk_ids
          LIMIT $limit
        }}
        UNION ALL
        WITH left_candidates, right_candidates
        WHERE left_candidates <> 1 OR right_candidates <> 1
        RETURN left_candidates AS left_candidates,
               right_candidates AS right_candidates,
               NULL AS target,
               [] AS left_fact_ids,
               [] AS right_fact_ids,
               [] AS left_source_chunk_ids,
               [] AS right_source_chunk_ids
        LIMIT 1
        """

        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {
                    "left": left,
                    "right": right,
                    "left_preds": left_preds,
                    "right_preds": right_preds,
                    "limit": limit,
                    "left_type": left_type,
                    "right_type": right_type,
                },
                access_scope=request.access_scope,
            )

        if rows:
            row0 = rows[0] or {}
            left_candidates = int(row0.get("left_candidates") or 1)
            right_candidates = int(row0.get("right_candidates") or 1)
            if left_candidates != 1 or right_candidates != 1:
                return ToolResult(
                    summary=(
                        "Intersection aborted due to ambiguous entities. "
                        f"left_candidates={left_candidates} right_candidates={right_candidates}. "
                        "Provide left_type/right_type to disambiguate."
                    ),
                    diagnostics={
                        "left": left,
                        "right": right,
                        "left_candidates": left_candidates,
                        "right_candidates": right_candidates,
                        "left_type": left_type or None,
                        "right_type": right_type or None,
                    },
                )

        targets = []
        evidences: List[EvidenceChunk] = []
        for idx, row in enumerate(rows or []):
            target = str((row or {}).get("target") or "").strip()
            if not target:
                continue
            targets.append(target)
            content = f"intersection_target: {left} & {right} -> {target}"
            chunk_id = derived_chunk_id(
                tool_name=self.descriptor.name, plan_step=request.plan_step, label=f"inter_{idx}", content=content
            )
            evidences.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source=self.descriptor.name,
                    content=content,
                    provenance={
                        "left": left,
                        "right": right,
                        "target": target,
                        "left_fact_ids": (row or {}).get("left_fact_ids") or [],
                        "right_fact_ids": (row or {}).get("right_fact_ids") or [],
                        "left_source_chunk_ids": (row or {}).get("left_source_chunk_ids") or [],
                        "right_source_chunk_ids": (row or {}).get("right_source_chunk_ids") or [],
                        "left_direction": left_direction,
                        "right_direction": right_direction,
                        "direction_forced_left": left_forced,
                        "direction_forced_right": right_forced,
                        "direction_forced_undirected_left": left_forced_undirected,
                        "direction_forced_undirected_right": right_forced_undirected,
                        "left_predicates": left_preds,
                        "right_predicates": right_preds,
                    },
                )
            )

        if not targets:
            return ToolResult(
                summary="Intersection query executed but found no shared targets under the given predicate filters.",
                diagnostics={
                    "left": left,
                    "right": right,
                    "left_direction": left_direction,
                    "right_direction": right_direction,
                    "direction_forced_left": left_forced,
                    "direction_forced_right": right_forced,
                    "direction_forced_undirected_left": left_forced_undirected,
                    "direction_forced_undirected_right": right_forced_undirected,
                },
            )
        return ToolResult(
            summary=f"Intersection query found {len(targets)} shared targets.",
            evidences=evidences,
            diagnostics={
                "left": left,
                "right": right,
                "targets": targets[: min(10, len(targets))],
                "left_direction": left_direction,
                "right_direction": right_direction,
                "direction_forced_left": left_forced,
                "direction_forced_right": right_forced,
                "direction_forced_undirected_left": left_forced_undirected,
                "direction_forced_undirected_right": right_forced_undirected,
            },
        )

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphIntersectionTool requires a GraphDeepSearchAdapter instance")
        return adapter

    @staticmethod
    def _predicate_filters(*, left_alias: str, right_alias: str, left_preds: Sequence[str], right_preds: Sequence[str]) -> str:
        filters: List[str] = []
        if left_preds:
            filters.append(f"{left_alias}.predicate IN $left_preds")
        if right_preds:
            filters.append(f"{right_alias}.predicate IN $right_preds")
        if not filters:
            return ""
        return "AND " + " AND ".join(filters)
