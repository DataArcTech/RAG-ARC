"""Temporal deterministic tools backed by Neo4j Cypher."""
import re

from encapsulation.data_model.deepsearch import EvidenceChunk

from core.graph_adapter.cypher import adapter_supports_cypher
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from .graph_ops_common import normalize_entity_name, normalize_predicates


_SAFE_CYPHER_PROPERTY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class GraphLatestTruthTool(GraphTool):
    """Deterministic 'latest truth' resolution over temporal facts."""

    descriptor = ToolDescriptor(
        name="graph.latest_truth",
        channel="graph",
        description="Deterministic latest-truth selection backed by Neo4j Cypher (orders by temporal attributes).",
        speed="fast",
        cost="low",
        strategy_tags=("temporal", "latest_truth", "deterministic"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_latest_truth",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "topic": {"type": "string", "description": "Topic entity name."},
                "topic_type": {"type": "string", "description": "Optional topic entity_type for disambiguation."},
                "predicates": {"type": "array", "items": {"type": "string"}, "description": "Predicates linking topic -> value."},
                "time_property": {"type": "string", "description": "Relationship property name used for ordering (optional)."},
            },
            required_extra_fields=("topic",),
        ),
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Latest truth requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        topic = normalize_entity_name(request.extra.get("topic"))
        if not topic:
            return ToolResult(summary="Latest truth requires a non-empty topic.")

        topic_type = str(request.extra.get("topic_type") or "").strip()
        predicates = normalize_predicates(request.extra.get("predicates"))
        time_property = str(request.extra.get("time_property") or "").strip()
        if time_property and not _SAFE_CYPHER_PROPERTY_RE.match(time_property):
            # Prevent Cypher injection by refusing non-identifier property names.
            time_property = ""

        predicate_clause = "AND r.predicate IN $predicates" if predicates else ""
        if time_property:
            order_expr = f"COALESCE(r.{time_property}, r.updated_at, r.created_at)"
        else:
            order_expr = "COALESCE(r.valid_from, r.effective_date, r.updated_at, r.created_at)"

        cypher = f"""
        MATCH (t0:Entity)
        WHERE COALESCE(t0.owner_id, $global_owner) = $owner_id
          AND t0.entity_name_normalized = $topic
          AND ($topic_type = '' OR t0.entity_type = $topic_type)
        WITH collect(t0) AS candidates
        WITH size(candidates) AS candidate_count,
             CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS t
        MATCH (t)-[r:RELATES_TO]->(v:Entity)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
          AND COALESCE(v.owner_id, $global_owner) = $owner_id
          {predicate_clause}
        RETURN candidate_count AS candidate_count,
               v.entity_name AS value,
               r.predicate AS predicate,
               {order_expr} AS sort_key,
               r.fact_id AS fact_id,
               r.source_chunk_ids AS source_chunk_ids
        ORDER BY sort_key DESC
        LIMIT 1
        """
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"topic": topic, "predicates": predicates, "topic_type": topic_type},
                access_scope=request.access_scope,
            )

        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        candidate_count = int((row0 or {}).get("candidate_count") or 1)
        if candidate_count != 1:
            return ToolResult(
                summary=(
                    "Latest truth aborted due to ambiguous topic entity name. "
                    f"candidate_count={candidate_count}. Provide topic_type to disambiguate."
                ),
                diagnostics={"topic": topic, "candidate_count": candidate_count, "topic_type": topic_type or None},
            )
        value = str((row0 or {}).get("value") or "").strip()
        if not value:
            return ToolResult(summary="Latest truth query returned no candidate values.", diagnostics={"topic": topic, "predicates": predicates})

        summary = f"latest_truth: topic={topic} value={value}"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="latest", content=summary)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=summary,
            provenance={
                "topic": topic,
                "predicates": predicates,
                "value": value,
                "predicate": (row0 or {}).get("predicate"),
                "sort_key": (row0 or {}).get("sort_key"),
                "fact_id": (row0 or {}).get("fact_id"),
                "source_chunk_ids": (row0 or {}).get("source_chunk_ids") or [],
                "time_property": time_property or None,
            },
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"value": value})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphLatestTruthTool requires a GraphDeepSearchAdapter instance")
        return adapter
