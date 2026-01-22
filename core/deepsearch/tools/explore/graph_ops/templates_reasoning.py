"""Deterministic reasoning templates (intersection, aggregation, set ops, rule checks)."""
from typing import Any, Dict, List, Mapping

from core.deepsearch.tools.base import ToolResult, ToolRunRequest
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DIAGNOSTIC, EVIDENCE_KIND_DERIVED
from core.graph_adapter.cypher import adapter_supports_cypher

from config.core.deepsearch.tool_defaults import (
    GRAPH_OPS_AGGREGATE_DEFAULT_LIMIT,
    GRAPH_OPS_AGGREGATE_MAX_LIMIT,
    GRAPH_OPS_INTERSECTION_DEFAULT_LIMIT,
    GRAPH_OPS_INTERSECTION_MAX_LIMIT,
    GRAPH_OPS_RULE_CHECK_DEFAULT_LIMIT,
    GRAPH_OPS_RULE_CHECK_MAX_LIMIT,
    GRAPH_OPS_SET_DIFFERENCE_DEFAULT_LIMIT,
    GRAPH_OPS_SET_DIFFERENCE_MAX_LIMIT,
)
from .graph_ops_common import (
    directionality_config,
    enforce_direction_for_sensitive_predicates,
    enforce_undirected_for_non_sensitive_predicates,
    limit_int,
    normalize_entity_name,
    normalize_predicates,
    rel_pattern,
)
from .templates_utils import build_derived_evidence, normalize_string_list


async def run_intersection(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops intersection requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    raw_left = str(args.get("left") or "").strip()
    raw_right = str(args.get("right") or "").strip()
    left = normalize_entity_name(raw_left)
    right = normalize_entity_name(raw_right)
    if not left or not right:
        return ToolResult(summary="intersection requires non-empty left/right entity names.")

    left_type = str(args.get("left_type") or "").strip()
    right_type = str(args.get("right_type") or "").strip()
    left_preds = normalize_predicates(args.get("left_predicates"))
    right_preds = normalize_predicates(args.get("right_predicates"))
    direction_default = str(args.get("direction") or "out")
    left_direction_raw = str(args.get("left_direction") or direction_default)
    right_direction_raw = str(args.get("right_direction") or direction_default)
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
    limit = limit_int(
        args.get("limit"),
        GRAPH_OPS_INTERSECTION_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_INTERSECTION_MAX_LIMIT,
    )

    rel_left = rel_pattern(left_direction, rel_var="lr", rel_type="RELATES_TO")
    rel_right = rel_pattern(right_direction, rel_var="rr", rel_type="RELATES_TO")
    predicate_clause = _intersection_predicate_clause(left_preds, right_preds)
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
      {predicate_clause}
      RETURN t.entity_name AS target,
             collect(DISTINCT lr.fact_id) AS left_fact_ids,
             collect(DISTINCT rr.fact_id) AS right_fact_ids,
             collect(DISTINCT lr.source_chunk_ids) AS left_source_chunk_ids,
             collect(DISTINCT rr.source_chunk_ids) AS right_source_chunk_ids
      LIMIT $limit
      UNION ALL
      WITH left_candidates, right_candidates
      WITH left_candidates, right_candidates
      WHERE left_candidates <> 1 OR right_candidates <> 1
      RETURN NULL AS target,
             [] AS left_fact_ids,
             [] AS right_fact_ids,
             [] AS left_source_chunk_ids,
             [] AS right_source_chunk_ids
      LIMIT 1
    }}
    RETURN left_candidates AS left_candidates,
           right_candidates AS right_candidates,
           target AS target,
           left_fact_ids AS left_fact_ids,
           right_fact_ids AS right_fact_ids,
           left_source_chunk_ids AS left_source_chunk_ids,
           right_source_chunk_ids AS right_source_chunk_ids
    """
    rows = await tool._acypher(
        request,
        cypher,
        {
            "left": left,
            "right": right,
            "left_type": left_type,
            "right_type": right_type,
            "left_predicates": left_preds,
            "right_predicates": right_preds,
            "limit": limit,
        },
    )

    row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
    left_candidates = int((row0 or {}).get("left_candidates") or 1)
    right_candidates = int((row0 or {}).get("right_candidates") or 1)
    if left_candidates != 1 or right_candidates != 1:
        return ToolResult(
            summary=(
                "intersection aborted due to ambiguous entities. "
                f"left_candidates={left_candidates} right_candidates={right_candidates}. "
                "Provide left_type/right_type to disambiguate."
            ),
            diagnostics={
                "left": left,
                "right": right,
                "left_candidates": left_candidates,
                "right_candidates": right_candidates,
            },
        )

    intersections: List[Dict[str, Any]] = []
    for row in rows or []:
        target = str((row or {}).get("target") or "").strip()
        if not target:
            continue
        intersections.append(
            {
                "target": target,
                "left_fact_ids": (row or {}).get("left_fact_ids") or [],
                "right_fact_ids": (row or {}).get("right_fact_ids") or [],
                "left_source_chunk_ids": (row or {}).get("left_source_chunk_ids") or [],
                "right_source_chunk_ids": (row or {}).get("right_source_chunk_ids") or [],
            }
        )
    summary = f"intersection: left={left} right={right} shared={len(intersections)}"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="intersection",
        content=summary,
        provenance={
            "left": left,
            "right": right,
            "left_predicates": left_preds,
            "right_predicates": right_preds,
            "left_direction": left_direction,
            "right_direction": right_direction,
            "left_direction_forced_sensitive": left_forced,
            "left_direction_forced_undirected": left_forced_undirected,
            "right_direction_forced_sensitive": right_forced,
            "right_direction_forced_undirected": right_forced_undirected,
            "intersections": intersections,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"targets": intersections[:50]})


async def run_set_difference(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops set_difference requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    excludes = [normalize_entity_name(x) for x in normalize_string_list(args.get("exclude")) if normalize_entity_name(x)]
    if not excludes:
        return ToolResult(summary="set_difference requires at least one exclude entity.")

    predicates = normalize_predicates(args.get("predicates"))
    direction_raw = str(args.get("direction") or "out")
    directionality = directionality_config(adapter)
    direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
        direction_raw, predicates, directionality=directionality, default_direction="out"
    )
    direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
        direction, predicates, directionality=directionality
    )
    limit = limit_int(
        args.get("limit"),
        GRAPH_OPS_SET_DIFFERENCE_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_SET_DIFFERENCE_MAX_LIMIT,
    )

    universe_entities = [
        normalize_entity_name(x)
        for x in normalize_string_list(args.get("universe_entities"))
        if normalize_entity_name(x)
    ]
    universe_type = str(args.get("universe_type") or "").strip()
    if not universe_entities and not universe_type:
        return ToolResult(summary="set_difference requires either universe_entities or universe_type.")

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

    rows = await tool._acypher(request, cypher, params)
    kept = [
        str((row or {}).get("entity") or "").strip()
        for row in rows or []
        if str((row or {}).get("entity") or "").strip()
    ]
    if not kept:
        return ToolResult(summary="set_difference query returned no entities.")

    content = f"set_difference: kept={len(kept)} exclude={excludes}"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="result",
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
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(
        summary=f"set_difference kept {len(kept)} entities.",
        evidences=[evidence],
        diagnostics={"kept": kept[:50]},
    )


async def run_aggregate(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops aggregate requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    raw_entity = str(args.get("entity") or "").strip()
    entity = normalize_entity_name(raw_entity)
    if not entity:
        return ToolResult(summary="aggregate requires a non-empty entity name.")

    entity_type = str(args.get("entity_type") or "").strip()
    predicate_list = normalize_predicates(args.get("predicate"))
    predicate = predicate_list[0] if predicate_list else None
    direction_raw = str(args.get("direction") or "out")
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
    limit = limit_int(
        args.get("limit"),
        GRAPH_OPS_AGGREGATE_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_AGGREGATE_MAX_LIMIT,
    )

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
    rows = await tool._acypher(
        request,
        cypher,
        {"entity": entity, "predicate": predicate, "limit": limit, "entity_type": entity_type},
    )

    row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
    candidate_count = int((row0 or {}).get("candidate_count") or 1)
    if candidate_count != 1:
        return ToolResult(
            summary=(
                "aggregate aborted due to ambiguous entity name. "
                f"candidate_count={candidate_count}. Provide entity_type to disambiguate."
            ),
            diagnostics={"entity": entity, "candidate_count": candidate_count, "entity_type": entity_type or None},
        )

    count = int((row0 or {}).get("distinct_count") or 0)
    examples = (row0 or {}).get("examples") or []
    summary = f"aggregate: entity={entity} predicate={predicate or '*'} direction={direction} distinct_count={count}"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="agg",
        content=summary,
        provenance={
            "distinct_count": count,
            "examples": examples,
            "entity": entity,
            "predicate": predicate,
            "direction": direction,
            "direction_forced_sensitive": forced_sensitive,
            "direction_forced_undirected": forced_undirected,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"distinct_count": count, "examples": examples})


async def run_rule_check(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops rule_check requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    conditions = args.get("conditions") or []
    if not isinstance(conditions, list) or not conditions:
        return ToolResult(summary="rule_check requires a non-empty conditions list.")
    limit = limit_int(
        args.get("limit"),
        GRAPH_OPS_RULE_CHECK_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_RULE_CHECK_MAX_LIMIT,
    )

    results: List[Dict[str, Any]] = []
    evidences = []
    all_ok = True
    invalid_indices: List[int] = []

    directionality = directionality_config(adapter)
    for idx, cond in enumerate(conditions):
        if not isinstance(cond, Mapping):
            invalid_indices.append(idx)
            all_ok = False
            continue
        head = normalize_entity_name(cond.get("head"))
        head_type = str(cond.get("head_type") or "").strip()
        tail = normalize_entity_name(cond.get("tail"))
        tail_type = str(cond.get("tail_type") or "").strip()
        predicate_list = normalize_predicates(cond.get("predicate"))
        predicate = predicate_list[0] if predicate_list else None
        direction_raw = str(cond.get("direction") or "out")
        direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
            direction_raw,
            [predicate] if predicate else [],
            directionality=directionality,
            default_direction="out",
        )
        direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            direction,
            [predicate] if predicate else [],
            directionality=directionality,
        )
        if not head or not tail or not predicate:
            invalid_indices.append(idx)
            all_ok = False
            continue

        rel = rel_pattern(direction, rel_var="r", rel_type="RELATES_TO")
        cypher = f"""
        MATCH (h0:Entity)
        WHERE COALESCE(h0.owner_id, $global_owner) = $owner_id
          AND h0.entity_name_normalized = $head
          AND ($head_type = '' OR h0.entity_type = $head_type)
        WITH collect(h0) AS head_nodes
        WITH size(head_nodes) AS head_candidates,
             CASE WHEN size(head_nodes) = 1 THEN head_nodes[0] ELSE NULL END AS h
        MATCH (t0:Entity)
        WHERE COALESCE(t0.owner_id, $global_owner) = $owner_id
          AND t0.entity_name_normalized = $tail
          AND ($tail_type = '' OR t0.entity_type = $tail_type)
        WITH head_candidates, h, collect(t0) AS tail_nodes
        WITH head_candidates,
             size(tail_nodes) AS tail_candidates,
             h,
             CASE WHEN size(tail_nodes) = 1 THEN tail_nodes[0] ELSE NULL END AS t
        OPTIONAL MATCH (h){rel}(t)
        WHERE head_candidates = 1
          AND tail_candidates = 1
          AND COALESCE(r.owner_id, $global_owner) = $owner_id
          AND r.predicate = $predicate
        RETURN head_candidates AS head_candidates,
               tail_candidates AS tail_candidates,
               r.fact_id AS fact_id,
               r.source_chunk_ids AS source_chunk_ids,
               r.text AS text
        LIMIT $limit
        """
        rows = await tool._acypher(
            request,
            cypher,
            {
                "head": head,
                "tail": tail,
                "predicate": predicate,
                "limit": limit,
                "head_type": head_type,
                "tail_type": tail_type,
            },
        )
        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        head_candidates = int((row0 or {}).get("head_candidates") or 1)
        tail_candidates = int((row0 or {}).get("tail_candidates") or 1)
        if head_candidates != 1 or tail_candidates != 1:
            ok = False
            rows = []
        else:
            ok = any(bool((row or {}).get("fact_id")) for row in (rows or []))
        all_ok = all_ok and ok
        results.append(
            {
                "condition": {
                    "head": head,
                    "head_type": head_type or None,
                    "predicate": predicate,
                    "tail": tail,
                    "tail_type": tail_type or None,
                    "direction": direction,
                    "direction_forced_sensitive": forced_sensitive,
                    "direction_forced_undirected": forced_undirected,
                    "head_candidates": head_candidates,
                    "tail_candidates": tail_candidates,
                },
                "ok": ok,
                "matches": rows or [],
            }
        )
        content = f"rule_condition[{idx}] ok={ok}: {head} -[{predicate}]-> {tail}"
        evidence = build_derived_evidence(
            tool_name=tool.descriptor.name,
            plan_step=request.plan_step,
            label=f"cond_{idx}",
            content=content,
            provenance={
                "condition": results[-1]["condition"],
                "matches": rows or [],
            },
            kind=EVIDENCE_KIND_DIAGNOSTIC if not ok else EVIDENCE_KIND_DERIVED,
        )
        evidences.append(evidence)

    summary = "rule_check ok" if all_ok else "rule_check failed"
    diagnostics = {
        "ok": all_ok,
        "results": results,
        "invalid_indices": invalid_indices,
    }
    return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics)


def _intersection_predicate_clause(left_preds: List[str], right_preds: List[str]) -> str:
    clauses = []
    if left_preds:
        clauses.append("AND lr.predicate IN $left_predicates")
    if right_preds:
        clauses.append("AND rr.predicate IN $right_predicates")
    return "\n".join(clauses)
