"""SDF schema templates."""
from typing import Any, Dict, List

from core.deepsearch.tools.base import ToolResult, ToolRunRequest
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.graph_adapter.cypher import adapter_supports_cypher

from config.core.deepsearch.tool_defaults import (
    GRAPH_OPS_SDF_DEFAULT_LIMIT,
    GRAPH_OPS_SDF_MAX_LIMIT,
)
from .graph_ops_common import normalize_entity_name
from .templates_utils import build_derived_evidence, unique_strings


def _limit_sdf(value: Any) -> int:
    try:
        limit = int(value) if value is not None else GRAPH_OPS_SDF_DEFAULT_LIMIT
    except (TypeError, ValueError):
        limit = GRAPH_OPS_SDF_DEFAULT_LIMIT
    return max(1, min(GRAPH_OPS_SDF_MAX_LIMIT, limit))


async def run_sdf_children(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops sdf_children requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    event = normalize_entity_name(args.get("event"))
    if not event:
        return ToolResult(summary="sdf_children requires a non-empty event name.")

    doc_namespace = str(args.get("doc_namespace") or "").strip()
    limit = _limit_sdf(args.get("limit"))

    cypher = """
    MATCH (t0:SDFEvent)
    WHERE COALESCE(t0.owner_id, $global_owner) = $owner_id
      AND t0.name_normalized = $event
      AND ($doc_namespace = '' OR t0.doc_namespace = $doc_namespace)
    WITH collect(t0) AS candidates
    WITH size(candidates) AS candidate_count,
         CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS t
    MATCH (t)-[r:SDF_HAS_SUBEVENT]->(c:SDFEvent)
    WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
      AND COALESCE(c.owner_id, $global_owner) = $owner_id
    RETURN candidate_count AS candidate_count,
           t.children_gate AS gate,
           c.name AS child,
           c.sdf_event_id AS child_event_id,
           r.importance AS importance,
           r.source_chunk_ids AS source_chunk_ids
    ORDER BY importance DESC, child ASC
    LIMIT $limit
    """
    rows = await tool._acypher(
        request,
        cypher,
        {"event": event, "doc_namespace": doc_namespace, "limit": limit},
    )

    row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
    candidate_count = int((row0 or {}).get("candidate_count") or 0)
    if candidate_count != 1:
        return ToolResult(
            summary=(
                "sdf_children aborted due to ambiguous or missing event name. "
                f"candidate_count={candidate_count}. Provide doc_namespace to disambiguate."
            ),
            diagnostics={"event": event, "candidate_count": candidate_count, "doc_namespace": doc_namespace or None},
        )

    gate = str((row0 or {}).get("gate") or "").strip() or None
    children = []
    source_chunk_ids: list[str] = []
    for row in rows or []:
        child = str((row or {}).get("child") or "").strip()
        if not child:
            continue
        children.append(
            {
                "child": child,
                "child_event_id": (row or {}).get("child_event_id"),
                "importance": (row or {}).get("importance"),
            }
        )
        source_chunk_ids.extend(list((row or {}).get("source_chunk_ids") or []))

    summary = f"sdf_children: event={event} gate={gate or 'unknown'} children={len(children)}"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="children",
        content=summary,
        provenance={
            "event": event,
            "doc_namespace": doc_namespace or None,
            "gate": gate,
            "children": children,
            "source_chunk_ids": unique_strings(source_chunk_ids),
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"children": children, "gate": gate})


async def run_sdf_dependencies(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops sdf_dependencies requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    event = normalize_entity_name(args.get("event"))
    if not event:
        return ToolResult(summary="sdf_dependencies requires a non-empty event name.")

    doc_namespace = str(args.get("doc_namespace") or "").strip()
    limit = _limit_sdf(args.get("limit"))

    cypher = """
    MATCH (t0:SDFEvent)
    WHERE COALESCE(t0.owner_id, $global_owner) = $owner_id
      AND t0.name_normalized = $event
      AND ($doc_namespace = '' OR t0.doc_namespace = $doc_namespace)
    WITH collect(t0) AS candidates
    WITH size(candidates) AS candidate_count,
         CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS t
    OPTIONAL MATCH (prev:SDFEvent)-[rb:SDF_BEFORE]->(t)
    WHERE COALESCE(rb.owner_id, $global_owner) = $owner_id
      AND COALESCE(prev.owner_id, $global_owner) = $owner_id
    WITH t, candidate_count,
         [x IN collect(DISTINCT CASE
             WHEN prev IS NULL THEN NULL
             ELSE {name: prev.name, event_id: prev.sdf_event_id, source_chunk_ids: rb.source_chunk_ids}
         END) WHERE x IS NOT NULL][..$limit] AS before_list
    OPTIONAL MATCH (t)-[ra:SDF_BEFORE]->(nxt:SDFEvent)
    WHERE COALESCE(ra.owner_id, $global_owner) = $owner_id
      AND COALESCE(nxt.owner_id, $global_owner) = $owner_id
    WITH candidate_count, before_list,
         [x IN collect(DISTINCT CASE
             WHEN nxt IS NULL THEN NULL
             ELSE {name: nxt.name, event_id: nxt.sdf_event_id, source_chunk_ids: ra.source_chunk_ids}
         END) WHERE x IS NOT NULL][..$limit] AS after_list
    RETURN candidate_count AS candidate_count,
           before_list AS before,
           after_list AS after
    """
    rows = await tool._acypher(
        request,
        cypher,
        {"event": event, "doc_namespace": doc_namespace, "limit": limit},
    )

    row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
    candidate_count = int((row0 or {}).get("candidate_count") or 0)
    if candidate_count != 1:
        return ToolResult(
            summary=(
                "sdf_dependencies aborted due to ambiguous or missing event name. "
                f"candidate_count={candidate_count}. Provide doc_namespace to disambiguate."
            ),
            diagnostics={"event": event, "candidate_count": candidate_count, "doc_namespace": doc_namespace or None},
        )

    before_list = (row0 or {}).get("before") or []
    after_list = (row0 or {}).get("after") or []
    summary = f"sdf_dependencies: event={event} before={len(before_list)} after={len(after_list)}"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="dependencies",
        content=summary,
        provenance={
            "event": event,
            "doc_namespace": doc_namespace or None,
            "before": before_list,
            "after": after_list,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"before": before_list, "after": after_list})
