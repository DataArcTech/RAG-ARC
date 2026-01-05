"""Deterministic aggregation tool backed by Neo4j Cypher."""
from encapsulation.data_model.deepsearch import EvidenceChunk

from core.graph_adapter.cypher import adapter_supports_cypher
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


class GraphAggregateTool(GraphTool):
    """Deterministic graph aggregation (COUNT DISTINCT)."""

    descriptor = ToolDescriptor(
        name="graph.aggregate",
        channel="graph",
        description="Deterministic aggregation (COUNT DISTINCT) over graph edges backed by Neo4j Cypher.",
        speed="fast",
        cost="low",
        strategy_tags=("aggregate", "count", "deterministic"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_aggregate",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "entity": {"type": "string", "description": "Anchor entity name."},
                "entity_type": {"type": "string", "description": "Optional entity_type for disambiguation."},
                "predicate": {"type": "string", "description": "Predicate to traverse/aggregate on."},
                "direction": {"type": "string", "enum": ["out", "in", "both"], "description": "Direction for traversal."},
                "limit": {"type": "integer", "minimum": 1, "description": "Max example neighbors returned."},
            },
            required_extra_fields=("entity",),
        ),
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Aggregation requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        entity = normalize_entity_name(request.extra.get("entity"))
        if not entity:
            return ToolResult(summary="Aggregation requires a non-empty entity name.")

        entity_type = str(request.extra.get("entity_type") or "").strip()
        predicate_list = normalize_predicates(request.extra.get("predicate"))
        predicate = predicate_list[0] if predicate_list else None
        direction_raw = str(request.extra.get("direction") or "out")
        directionality = directionality_config(adapter)
        direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
            direction_raw,
            [predicate] if predicate else [],
            directionality=directionality,
            default_direction="out",
        )
        direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            direction, [predicate] if predicate else [], directionality=directionality
        )
        limit = limit_int(request.extra.get("limit"), 10, max_value=50)

        rel = rel_pattern(direction, rel_var="rel", rel_type="RELATES_TO")
        pred_clause = "AND rel.predicate = $predicate" if predicate else ""
        cypher = f"""
        MATCH (e0:Entity)
        WHERE COALESCE(e0.owner_id, $global_owner) = $owner_id
          AND e0.entity_name_normalized = $entity
          AND ($entity_type = '' OR e0.entity_type = $entity_type)
        WITH collect(e0) AS candidates
        WITH size(candidates) AS candidate_count,
             CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS e
        MATCH (e){rel}(n:Entity)
        WHERE COALESCE(rel.owner_id, $global_owner) = $owner_id
          AND COALESCE(n.owner_id, $global_owner) = $owner_id
          {pred_clause}
        RETURN candidate_count AS candidate_count,
               count(DISTINCT COALESCE(n.entity_canonical_key, n.entity_canonical_name, n.entity_name, n.entity_id)) AS distinct_count,
               collect(DISTINCT COALESCE(n.entity_canonical_name, n.entity_name))[..$limit] AS examples
        """

        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"entity": entity, "predicate": predicate, "limit": limit, "entity_type": entity_type},
                access_scope=request.access_scope,
            )

        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        candidate_count = int((row0 or {}).get("candidate_count") or 1)
        if candidate_count != 1:
            return ToolResult(
                summary=(
                    "Aggregation aborted due to ambiguous entity name. "
                    f"candidate_count={candidate_count}. Provide entity_type to disambiguate."
                ),
                diagnostics={"entity": entity, "candidate_count": candidate_count, "entity_type": entity_type or None},
            )
        count = int((row0 or {}).get("distinct_count") or 0)
        examples = (row0 or {}).get("examples") or []

        content = f"aggregate: entity={entity} predicate={predicate or '*'} direction={direction} distinct_count={count}"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="agg", content=content)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=content,
            provenance={
                "distinct_count": count,
                "examples": examples,
                "entity": entity,
                "predicate": predicate,
                "direction": direction,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
            },
        )
        return ToolResult(
            summary=content,
            evidences=[evidence],
            diagnostics={
                "distinct_count": count,
                "examples": examples,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
            },
        )

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphAggregateTool requires a GraphDeepSearchAdapter instance")
        return adapter
