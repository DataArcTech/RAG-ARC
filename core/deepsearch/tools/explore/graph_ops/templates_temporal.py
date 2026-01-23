"""Temporal templates (latest-truth)."""
import re
from typing import Any, Dict

from core.deepsearch.entity_resolution import build_default_entity_resolver
from core.deepsearch.tools.base import ToolResult, ToolRunRequest
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.graph_adapter.cypher import adapter_supports_cypher

from config.core.deepsearch.tool_defaults import (
    GRAPH_OPS_LATEST_TRUTH_DEFAULT_LIMIT,
    NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN,
    NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN,
    NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT,
    NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS,
    NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN,
)
from .graph_ops_common import normalize_entity_name, normalize_predicates
from .templates_utils import build_derived_evidence


_SAFE_CYPHER_PROPERTY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _resolution_candidate_payload(candidate: Any) -> Dict[str, Any]:
    return {
        "entity_id": getattr(candidate, "entity_id", None),
        "entity_name": getattr(candidate, "entity_name", None),
        "entity_name_normalized": getattr(candidate, "entity_name_normalized", None),
        "entity_type": getattr(candidate, "entity_type", None),
        "entity_type_key": getattr(candidate, "entity_type_key", None),
        "strategy": getattr(candidate, "strategy", None),
        "hit_count": getattr(candidate, "hit_count", None),
        "edge_count": getattr(candidate, "edge_count", None),
        "mention_count": getattr(candidate, "mention_count", None),
        "faiss_score": getattr(candidate, "faiss_score", None),
        "score": getattr(candidate, "score", None),
        "score_breakdown": getattr(candidate, "score_breakdown", None),
    }


async def run_latest_truth(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops latest_truth requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    raw_topic = str(args.get("topic") or "").strip()
    topic = normalize_entity_name(raw_topic)
    if not topic:
        return ToolResult(summary="latest_truth requires a non-empty topic.")

    topic_type = str(args.get("topic_type") or "").strip()
    predicates = normalize_predicates(args.get("predicates"))
    time_property = str(args.get("time_property") or "").strip()
    resolution_diag = None
    if time_property and not _SAFE_CYPHER_PROPERTY_RE.match(time_property):
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
    LIMIT {GRAPH_OPS_LATEST_TRUTH_DEFAULT_LIMIT}
    """
    rows = await tool._acypher(
        request,
        cypher,
        {"topic": topic, "predicates": predicates, "topic_type": topic_type},
    )

    row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
    candidate_count = int((row0 or {}).get("candidate_count") or 1)
    if candidate_count != 1:
        resolver = build_default_entity_resolver(
            enabled=True,
            candidate_limit=NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT,
            min_token_len=NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN,
            min_token_hits=NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS,
            auto_score_min=NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN,
            auto_score_margin=NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN,
        )
        res = await resolver.resolve(
            adapter=adapter,
            access_scope=request.access_scope,
            raw_entity=raw_topic,
            entity_type_hint=topic_type,
        )
        if res.resolved_candidate is not None:
            resolution_diag = dict(res.diagnostics)
            cypher2 = f"""
            MATCH (t:Entity {{entity_id: $topic_id}})
            WHERE COALESCE(t.owner_id, $global_owner) = $owner_id
            MATCH (t)-[r:RELATES_TO]->(v:Entity)
            WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
              AND COALESCE(v.owner_id, $global_owner) = $owner_id
              {predicate_clause}
            RETURN v.entity_name AS value,
                   r.predicate AS predicate,
                   {order_expr} AS sort_key,
                   r.fact_id AS fact_id,
                   r.source_chunk_ids AS source_chunk_ids
            ORDER BY sort_key DESC
            LIMIT {GRAPH_OPS_LATEST_TRUTH_DEFAULT_LIMIT}
            """
            rows = await tool._acypher(
                request,
                cypher2,
                {"topic_id": res.resolved_candidate.entity_id, "predicates": predicates},
            )
            row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        else:
            return ToolResult(
                summary=(
                    "latest_truth aborted due to ambiguous topic entity name. "
                    f"candidate_count={candidate_count}. Provide topic_type to disambiguate."
                ),
                diagnostics={
                    "topic": topic,
                    "candidate_count": candidate_count,
                    "topic_type": topic_type or None,
                    "resolution_candidates": [_resolution_candidate_payload(c) for c in res.candidates],
                    "resolution_diagnostics": dict(res.diagnostics),
                },
            )

    value = str((row0 or {}).get("value") or "").strip()
    if not value:
        return ToolResult(summary="latest_truth query returned no candidate values.", diagnostics={"topic": topic, "predicates": predicates})

    summary = f"latest_truth: topic={topic} value={value}"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="latest",
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
            "resolution": resolution_diag,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"value": value})
