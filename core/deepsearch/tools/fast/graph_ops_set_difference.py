"""Deterministic set-difference tool backed by Neo4j Cypher."""
from typing import Any, Dict

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


class GraphSetDifferenceTool(GraphTool):
    """Set difference queries (e.g., 'which products do NOT contain X')."""

    descriptor = ToolDescriptor(
        name="graph.set_difference",
        channel="graph",
        description="Deterministic set-difference query backed by Neo4j Cypher.",
        speed="fast",
        cost="low",
        strategy_tags=("set_ops", "difference", "deterministic"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_set_difference",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "universe_type": {"type": "string", "description": "Entity type defining the universe (e.g. Product)."},
                "universe_entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit universe entities (names).",
                },
                "exclude": {"type": "array", "items": {"type": "string"}, "description": "Excluded entities (names)."},
                "predicates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Predicates linking universe -> exclude.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["out", "in", "both"],
                    "description": "Relation direction for exclusion test.",
                },
                "limit": {"type": "integer", "minimum": 1, "description": "Max items returned."},
            },
            required_extra_fields=("exclude",),
        ),
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not isinstance(adapter, GraphCypherQueryable) or not adapter.cypher_capable():
            return ToolResult(
                summary="Set difference requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        excludes_raw = request.extra.get("exclude") or []
        if isinstance(excludes_raw, str):
            excludes_raw = [excludes_raw]
        excludes = [normalize_entity_name(x) for x in excludes_raw if normalize_entity_name(x)]
        if not excludes:
            return ToolResult(summary="Set difference requires at least one exclude entity.")

        predicates = normalize_predicates(request.extra.get("predicates"))
        direction_raw = str(request.extra.get("direction") or "out")
        directionality = directionality_config(adapter)
        direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
            direction_raw, predicates, directionality=directionality, default_direction="out"
        )
        direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            direction, predicates, directionality=directionality
        )
        limit = limit_int(request.extra.get("limit"), 20, max_value=200)

        universe_entities_raw = request.extra.get("universe_entities") or []
        if isinstance(universe_entities_raw, str):
            universe_entities_raw = [universe_entities_raw]
        universe_entities = [normalize_entity_name(x) for x in universe_entities_raw if normalize_entity_name(x)]
        universe_type = str(request.extra.get("universe_type") or "").strip()

        if not universe_entities and not universe_type:
            return ToolResult(summary="Set difference requires either universe_entities or universe_type.")

        rel = rel_pattern(direction, rel_var="rel", rel_type="RELATES_TO")
        pred_clause = "AND rel.predicate IN $predicates" if predicates else ""

        if universe_entities:
            cypher = f"""
            UNWIND $universe AS uname
            MATCH (u:Entity)
            WHERE COALESCE(u.owner_id, $global_owner) = $owner_id AND u.entity_name_normalized = uname
            WITH collect(u) AS universe_nodes
            UNWIND universe_nodes AS u
            OPTIONAL MATCH (u){rel}(x:Entity)
            WHERE COALESCE(rel.owner_id, $global_owner) = $owner_id
              AND COALESCE(x.owner_id, $global_owner) = $owner_id
              AND x.entity_name_normalized IN $exclude
              {pred_clause}
            WITH u, count(rel) AS hit_count
            WHERE hit_count = 0
            RETURN u.entity_name AS entity
            LIMIT $limit
            """
            params: Dict[str, Any] = {
                "universe": universe_entities,
                "exclude": excludes,
                "predicates": predicates,
                "limit": limit,
            }
        else:
            cypher = f"""
            MATCH (u:Entity)
            WHERE COALESCE(u.owner_id, $global_owner) = $owner_id
              AND u.entity_type = $universe_type
            OPTIONAL MATCH (u){rel}(x:Entity)
            WHERE COALESCE(rel.owner_id, $global_owner) = $owner_id
              AND COALESCE(x.owner_id, $global_owner) = $owner_id
              AND x.entity_name_normalized IN $exclude
              {pred_clause}
            WITH u, count(rel) AS hit_count
            WHERE hit_count = 0
            RETURN u.entity_name AS entity
            LIMIT $limit
            """
            params = {"universe_type": universe_type, "exclude": excludes, "predicates": predicates, "limit": limit}

        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)

        kept = [
            str((row or {}).get("entity") or "").strip()
            for row in rows or []
            if str((row or {}).get("entity") or "").strip()
        ]
        if not kept:
            return ToolResult(summary="Set difference query returned no entities.")

        content = f"set_difference: kept={len(kept)} exclude={excludes}"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="result", content=content)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=content,
            provenance={
                "kept": kept,
                "exclude": excludes,
                "predicates": predicates,
                "direction": direction,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
                "universe_type": universe_type or None,
                "universe_entities": universe_entities or None,
            },
        )
        return ToolResult(
            summary=f"Set difference kept {len(kept)} entities.",
            evidences=[evidence],
            diagnostics={
                "kept": kept[:50],
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
            },
        )

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphSetDifferenceTool requires a GraphDeepSearchAdapter instance")
        return adapter
