"""Relation-path exploration/grounding templates."""
from typing import Any, Dict, List

from core.deepsearch.entity_resolution import build_default_entity_resolver
from core.deepsearch.tools.base import ToolResult, ToolRunRequest
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.graph_adapter.cypher import adapter_supports_cypher

from config.core.deepsearch.tool_defaults import (
    GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_HOPS,
    GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_PATHS,
    GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_SEQUENCES,
    GRAPH_OPS_REL_PATH_EXPLORE_MAX_HOPS,
    GRAPH_OPS_REL_PATH_EXPLORE_MAX_PATHS,
    GRAPH_OPS_REL_PATH_EXPLORE_MAX_SEQUENCES,
    GRAPH_OPS_REL_PATH_GROUND_DEFAULT_MAX_PATHS,
    GRAPH_OPS_REL_PATH_GROUND_MAX_PATHS,
    NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN,
    NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN,
    NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT,
    NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS,
    NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN,
)
from .graph_ops_common import (
    directionality_config,
    enforce_direction_for_sensitive_predicates,
    enforce_undirected_for_non_sensitive_predicates,
    limit_int,
    normalize_entity_name,
    normalize_predicates,
    rel_pattern_varlen,
)
from .templates_utils import build_derived_evidence, normalize_predicate_sequence


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


async def run_relation_path_explore(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops relation_path_explore requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    raw_entity = str(args.get("entity") or "").strip()
    entity = normalize_entity_name(raw_entity)
    if not entity:
        return ToolResult(summary="relation_path_explore requires a non-empty entity name.")

    predicates = normalize_predicates(args.get("predicates"))
    entity_type = str(args.get("entity_type") or "").strip()
    direction_raw = str(args.get("direction") or "out")
    directionality = directionality_config(adapter)
    direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
        direction_raw, predicates, directionality=directionality, default_direction="out"
    )
    direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
        direction, predicates, directionality=directionality
    )
    max_hops = limit_int(
        args.get("max_hops"),
        GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_HOPS,
        max_value=GRAPH_OPS_REL_PATH_EXPLORE_MAX_HOPS,
    )
    max_paths = limit_int(
        args.get("max_paths"),
        GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_PATHS,
        max_value=GRAPH_OPS_REL_PATH_EXPLORE_MAX_PATHS,
    )
    max_sequences = limit_int(
        args.get("max_sequences"),
        GRAPH_OPS_REL_PATH_EXPLORE_DEFAULT_MAX_SEQUENCES,
        max_value=GRAPH_OPS_REL_PATH_EXPLORE_MAX_SEQUENCES,
    )
    resolution_diag: Dict[str, Any] | None = None

    rel = rel_pattern_varlen(direction, rel_type="RELATES_TO", max_hops=max_hops)
    predicate_filter = "AND ALL(r IN relationships(p) WHERE r.predicate IN $predicates)" if predicates else ""
    cypher = f"""
    // relation_path_explore
    MATCH (s0:Entity)
    WHERE COALESCE(s0.owner_id, $global_owner) = $owner_id
      AND s0.entity_name_normalized = $entity
      AND ($entity_type = '' OR s0.entity_type = $entity_type)
    WITH collect(s0) AS candidates
    WITH size(candidates) AS candidate_count,
         CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS s
    MATCH p=(s){rel}(t:Entity)
    WHERE s IS NOT NULL
      AND COALESCE(t.owner_id, $global_owner) = $owner_id
      AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
      {predicate_filter}
    WITH candidate_count,
         [r IN relationships(p) | r.predicate] AS predicate_sequence,
         t.entity_name AS target_entity,
         [r IN relationships(p) | r.fact_id] AS fact_ids,
         [r IN relationships(p) | r.source_chunk_ids] AS source_chunk_ids
    LIMIT $max_paths
    WITH candidate_count,
         predicate_sequence,
         collect(DISTINCT target_entity)[0..5] AS targets,
         collect(fact_ids)[0..3] AS fact_ids_samples,
         collect(source_chunk_ids)[0..3] AS source_chunk_ids_samples,
         count(*) AS path_count
    RETURN candidate_count AS candidate_count,
           predicate_sequence AS predicate_sequence,
           targets AS targets,
           fact_ids_samples AS fact_ids_samples,
           source_chunk_ids_samples AS source_chunk_ids_samples,
           path_count AS path_count
    ORDER BY path_count DESC
    LIMIT $max_sequences
    """
    params = {
        "entity": entity,
        "entity_type": entity_type,
        "predicates": predicates,
        "max_paths": max_paths,
        "max_sequences": max_sequences,
    }
    rows = await tool._acypher(request, cypher, params)

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
            raw_entity=raw_entity,
            entity_type_hint=entity_type,
        )
        if res.resolved_candidate is not None:
            resolution_diag = dict(res.diagnostics)
            cypher2 = f"""
            // relation_path_explore (resolved by entity_id)
            MATCH (s:Entity {{entity_id: $entity_id}})
            WHERE COALESCE(s.owner_id, $global_owner) = $owner_id
            MATCH p=(s){rel}(t:Entity)
            WHERE COALESCE(t.owner_id, $global_owner) = $owner_id
              AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
              {predicate_filter}
            WITH [r IN relationships(p) | r.predicate] AS predicate_sequence,
                 t.entity_name AS target_entity,
                 [r IN relationships(p) | r.fact_id] AS fact_ids,
                 [r IN relationships(p) | r.source_chunk_ids] AS source_chunk_ids
            LIMIT $max_paths
            WITH predicate_sequence,
                 collect(DISTINCT target_entity)[0..5] AS targets,
                 collect(fact_ids)[0..3] AS fact_ids_samples,
                 collect(source_chunk_ids)[0..3] AS source_chunk_ids_samples,
                 count(*) AS path_count
            RETURN predicate_sequence AS predicate_sequence,
                   targets AS targets,
                   fact_ids_samples AS fact_ids_samples,
                   source_chunk_ids_samples AS source_chunk_ids_samples,
                   path_count AS path_count
            ORDER BY path_count DESC
            LIMIT $max_sequences
            """
            params2 = {
                "entity_id": res.resolved_candidate.entity_id,
                "predicates": predicates,
                "max_paths": max_paths,
                "max_sequences": max_sequences,
            }
            rows = await tool._acypher(request, cypher2, params2)
        else:
            return ToolResult(
                summary=(
                    "relation_path_explore aborted due to ambiguous entity. "
                    f"candidate_count={candidate_count}. Provide entity_type to disambiguate."
                ),
                diagnostics={
                    "entity": entity,
                    "candidate_count": candidate_count,
                    "entity_type": entity_type or None,
                    "resolution_candidates": [_resolution_candidate_payload(c) for c in res.candidates],
                    "resolution_diagnostics": dict(res.diagnostics),
                },
            )

    sequences: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        seq = row.get("predicate_sequence")
        if not isinstance(seq, list) or not seq:
            continue
        sequences.append(
            {
                "predicate_sequence": [str(item) for item in seq if str(item).strip()],
                "path_count": int(row.get("path_count") or 0),
                "targets": row.get("targets") if isinstance(row.get("targets"), list) else [],
                "fact_ids_samples": row.get("fact_ids_samples") if isinstance(row.get("fact_ids_samples"), list) else [],
                "source_chunk_ids_samples": row.get("source_chunk_ids_samples")
                if isinstance(row.get("source_chunk_ids_samples"), list)
                else [],
            }
        )

    summary = f"relation_path_explore: {entity} sequences={len(sequences)} (max_hops={max_hops})"
    content = f"relation_path_explore({entity}): {len(sequences)} sequences"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="relation_paths",
        content=content,
        provenance={
            "entity": entity,
            "entity_type": entity_type or None,
            "direction": direction,
            "direction_forced_sensitive": forced_sensitive,
            "direction_forced_undirected": forced_undirected,
            "max_hops": max_hops,
            "max_paths": max_paths,
            "max_sequences": max_sequences,
            "allowed_predicates": predicates,
            "resolution": resolution_diag,
            "relation_paths": sequences,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"sequence_count": len(sequences)})


async def run_relation_path_ground(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops relation_path_ground requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    raw_source = str(args.get("source") or "").strip()
    source = normalize_entity_name(raw_source)
    if not source:
        return ToolResult(summary="relation_path_ground requires a non-empty source entity name.")

    raw_seq = args.get("predicate_sequence")
    predicate_sequence = normalize_predicate_sequence(raw_seq)
    if not predicate_sequence:
        return ToolResult(summary="relation_path_ground requires a non-empty predicate_sequence.")

    source_type = str(args.get("source_type") or "").strip()
    direction_raw = str(args.get("direction") or "out")
    directionality = directionality_config(adapter)
    direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
        direction_raw, predicate_sequence, directionality=directionality, default_direction="out"
    )
    direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
        direction, predicate_sequence, directionality=directionality
    )
    max_paths = limit_int(
        args.get("max_paths"),
        GRAPH_OPS_REL_PATH_GROUND_DEFAULT_MAX_PATHS,
        max_value=GRAPH_OPS_REL_PATH_GROUND_MAX_PATHS,
    )
    hops = len(predicate_sequence)
    resolution_diag: Dict[str, Any] | None = None

    rel = rel_pattern_varlen(direction, rel_type="RELATES_TO", max_hops=hops)
    cypher = f"""
    // relation_path_ground
    MATCH (s0:Entity)
    WHERE COALESCE(s0.owner_id, $global_owner) = $owner_id
      AND s0.entity_name_normalized = $source
      AND ($source_type = '' OR s0.entity_type = $source_type)
    WITH collect(s0) AS source_nodes
    WITH size(source_nodes) AS candidate_count,
         CASE WHEN size(source_nodes) = 1 THEN source_nodes[0] ELSE NULL END AS s
    MATCH p=(s){rel}(t:Entity)
    WHERE s IS NOT NULL
      AND COALESCE(t.owner_id, $global_owner) = $owner_id
      AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
    WITH candidate_count, p, relationships(p) AS rels
    WHERE size(rels) = $hops
      AND all(i IN range(0, $hops - 1) WHERE rels[i].predicate = $predicates[i])
    RETURN candidate_count AS candidate_count,
           [n IN nodes(p) | n.entity_name] AS nodes,
           [r IN rels | r.predicate] AS predicates,
           [r IN rels | r.fact_id] AS fact_ids,
           [r IN rels | r.source_chunk_ids] AS source_chunk_ids
    LIMIT $max_paths
    """
    params = {
        "source": source,
        "source_type": source_type,
        "predicates": list(predicate_sequence),
        "hops": hops,
        "max_paths": max_paths,
    }
    rows = await tool._acypher(request, cypher, params)

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
            raw_entity=raw_source,
            entity_type_hint=source_type,
        )
        if res.resolved_candidate is not None:
            resolution_diag = dict(res.diagnostics)
            cypher2 = f"""
            // relation_path_ground (resolved by entity_id)
            MATCH (s:Entity {{entity_id: $source_id}})
            WHERE COALESCE(s.owner_id, $global_owner) = $owner_id
            MATCH p=(s){rel}(t:Entity)
            WHERE COALESCE(t.owner_id, $global_owner) = $owner_id
              AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
            WITH p, relationships(p) AS rels
            WHERE size(rels) = $hops
              AND all(i IN range(0, $hops - 1) WHERE rels[i].predicate = $predicates[i])
            RETURN [n IN nodes(p) | n.entity_name] AS nodes,
                   [r IN rels | r.predicate] AS predicates,
                   [r IN rels | r.fact_id] AS fact_ids,
                   [r IN rels | r.source_chunk_ids] AS source_chunk_ids
            LIMIT $max_paths
            """
            params2 = {
                "source_id": res.resolved_candidate.entity_id,
                "predicates": list(predicate_sequence),
                "hops": hops,
                "max_paths": max_paths,
            }
            rows = await tool._acypher(request, cypher2, params2)
        else:
            return ToolResult(
                summary=(
                    "relation_path_ground aborted due to ambiguous source entity. "
                    f"candidate_count={candidate_count}. Provide source_type to disambiguate."
                ),
                diagnostics={
                    "source": source,
                    "candidate_count": candidate_count,
                    "source_type": source_type or None,
                    "resolution_candidates": [_resolution_candidate_payload(c) for c in res.candidates],
                    "resolution_diagnostics": dict(res.diagnostics),
                },
            )

    grounded: List[Dict[str, Any]] = []
    frontier: List[str] = []
    seen_frontier: set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        nodes = row.get("nodes") if isinstance(row.get("nodes"), list) else []
        if nodes:
            leaf = str(nodes[-1])
            if leaf and leaf not in seen_frontier:
                seen_frontier.add(leaf)
                frontier.append(leaf)
        grounded.append(
            {
                "nodes": nodes,
                "predicates": row.get("predicates") if isinstance(row.get("predicates"), list) else [],
                "fact_ids": row.get("fact_ids") if isinstance(row.get("fact_ids"), list) else [],
                "source_chunk_ids": row.get("source_chunk_ids") if isinstance(row.get("source_chunk_ids"), list) else [],
            }
        )

    summary = f"relation_path_ground: {source} grounded_paths={len(grounded)} hops={hops}"
    content = f"relation_path_ground({source}): {len(grounded)} grounded paths"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="grounded_paths",
        content=content,
        provenance={
            "source": source,
            "source_type": source_type or None,
            "direction": direction,
            "direction_forced_sensitive": forced_sensitive,
            "direction_forced_undirected": forced_undirected,
            "resolution": resolution_diag,
            "predicate_sequence": list(predicate_sequence),
            "grounded_paths": grounded,
            "frontier_entities": frontier,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(
        summary=summary,
        evidences=[evidence],
        diagnostics={"grounded_path_count": len(grounded), "frontier_entity_count": len(frontier)},
    )
