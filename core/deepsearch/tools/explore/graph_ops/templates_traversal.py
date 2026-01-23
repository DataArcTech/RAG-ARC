"""Traversal-oriented graph.ops templates (path, neighbors, hierarchy)."""
from typing import Any, Dict, List

from core.deepsearch.entity_resolution import build_default_entity_resolver
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED, EVIDENCE_KIND_DIAGNOSTIC
from core.deepsearch.tools.base import ToolResult, ToolRunRequest
from core.graph_adapter.cypher import adapter_supports_cypher

from config.core.deepsearch.tool_defaults import (
    GRAPH_OPS_NEIGHBORS_DEFAULT_LIMIT,
    GRAPH_OPS_NEIGHBORS_MAX_LIMIT,
    GRAPH_OPS_PATH_EXISTS_DEFAULT_MAX_HOPS,
    GRAPH_OPS_PATH_EXISTS_MAX_HOPS,
    GRAPH_OPS_TRACE_TO_ROOT_DEFAULT_MAX_HOPS,
    GRAPH_OPS_TRACE_TO_ROOT_MAX_HOPS,
    NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN,
    NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN,
    NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT,
    NEIGHBORS_ENTITY_RESOLUTION_ENABLED,
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
    rel_pattern,
    rel_pattern_varlen,
)
from .templates_utils import build_derived_evidence, unique_strings


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


async def run_path_exists(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops path_exists requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    raw_source = str(args.get("source") or "").strip()
    raw_target = str(args.get("target") or "").strip()
    source = normalize_entity_name(raw_source)
    target = normalize_entity_name(raw_target)
    if not source or not target:
        return ToolResult(summary="path_exists requires non-empty source/target entity names.")

    predicates = normalize_predicates(args.get("predicates"))
    source_type = str(args.get("source_type") or "").strip()
    target_type = str(args.get("target_type") or "").strip()
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
        GRAPH_OPS_PATH_EXISTS_DEFAULT_MAX_HOPS,
        max_value=GRAPH_OPS_PATH_EXISTS_MAX_HOPS,
    )

    rel = rel_pattern_varlen(direction, rel_type="RELATES_TO", max_hops=max_hops)
    predicate_filter = "AND ALL(r IN relationships(p) WHERE r.predicate IN $predicates)" if predicates else ""
    cypher = f"""
    MATCH (s0:Entity)
    WHERE COALESCE(s0.owner_id, $global_owner) = $owner_id
      AND s0.entity_name_normalized = $source
      AND ($source_type = '' OR s0.entity_type = $source_type)
    WITH collect(s0) AS source_nodes
    WITH size(source_nodes) AS source_candidates,
         CASE WHEN size(source_nodes) = 1 THEN source_nodes[0] ELSE NULL END AS s
    MATCH (t0:Entity)
    WHERE COALESCE(t0.owner_id, $global_owner) = $owner_id
      AND t0.entity_name_normalized = $target
      AND ($target_type = '' OR t0.entity_type = $target_type)
    WITH source_candidates, s, collect(t0) AS target_nodes
    WITH source_candidates,
         size(target_nodes) AS target_candidates,
         s,
         CASE WHEN size(target_nodes) = 1 THEN target_nodes[0] ELSE NULL END AS t
    OPTIONAL MATCH p = (s){rel}(t)
    WHERE source_candidates = 1
      AND target_candidates = 1
      AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
      {predicate_filter}
    RETURN source_candidates AS source_candidates,
           target_candidates AS target_candidates,
           CASE WHEN p IS NULL THEN [] ELSE [n IN nodes(p) | n.entity_name] END AS nodes,
           CASE WHEN p IS NULL THEN [] ELSE [r IN relationships(p) | r.predicate] END AS predicates,
           CASE WHEN p IS NULL THEN [] ELSE [r IN relationships(p) | r.fact_id] END AS fact_ids,
           CASE WHEN p IS NULL THEN [] ELSE [r IN relationships(p) | r.source_chunk_ids] END AS source_chunk_ids
    ORDER BY COALESCE(length(p), 999) ASC
    LIMIT 1
    """
    params = {
        "source": source,
        "target": target,
        "predicates": predicates,
        "source_type": source_type,
        "target_type": target_type,
    }
    rows = await tool._acypher(request, cypher, params)

    row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
    source_candidates = int((row0 or {}).get("source_candidates") or 1)
    target_candidates = int((row0 or {}).get("target_candidates") or 1)
    if source_candidates != 1 or target_candidates != 1:
        return ToolResult(
            summary=(
                "path_exists aborted due to ambiguous entities. "
                f"source_candidates={source_candidates} target_candidates={target_candidates}. "
                "Provide source_type/target_type to disambiguate."
            ),
            diagnostics={
                "source": source,
                "target": target,
                "source_candidates": source_candidates,
                "target_candidates": target_candidates,
                "source_type": source_type or None,
                "target_type": target_type or None,
            },
        )

    nodes = (row0 or {}).get("nodes") or []
    ok = bool(nodes) and len(nodes) >= 2
    summary = f"path_exists ok={ok}: {source} -> {target} (max_hops={max_hops})"
    if not ok:
        return ToolResult(summary=summary, diagnostics={"ok": False, "source": source, "target": target, "max_hops": max_hops})

    content = "path: " + " -> ".join(str(n) for n in nodes)
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="path",
        content=content,
        provenance={
            "nodes": nodes,
            "predicates": (row0 or {}).get("predicates") or [],
            "fact_ids": (row0 or {}).get("fact_ids") or [],
            "source_chunk_ids": (row0 or {}).get("source_chunk_ids") or [],
            "direction": direction,
            "direction_forced_sensitive": forced_sensitive,
            "direction_forced_undirected": forced_undirected,
            "allowed_predicates": predicates,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"ok": True, "nodes": nodes})


async def run_neighbors(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops neighbors requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    raw_entity = str(args.get("entity") or "").strip()
    entity = normalize_entity_name(raw_entity)
    if not entity:
        return ToolResult(summary="neighbors requires a non-empty entity name.")

    entity_type = str(args.get("entity_type") or "").strip()
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
        GRAPH_OPS_NEIGHBORS_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_NEIGHBORS_MAX_LIMIT,
    )

    rel = rel_pattern(direction, rel_var="r", rel_type="RELATES_TO")
    predicate_clause = "AND r.predicate IN $predicates" if predicates else ""
    cypher = f"""
    MATCH (e0:Entity)
    WHERE COALESCE(e0.owner_id, $global_owner) = $owner_id
      AND e0.entity_name_normalized = $entity
      AND ($entity_type = '' OR e0.entity_type = $entity_type)
    WITH collect(e0) AS candidates
    WITH size(candidates) AS candidate_count,
         CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS e
    MATCH (e){rel}(n:Entity)
    WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
      AND COALESCE(n.owner_id, $global_owner) = $owner_id
      {predicate_clause}
    RETURN candidate_count AS candidate_count,
           n.entity_name AS neighbor,
           r.predicate AS predicate,
           r.fact_id AS fact_id,
           r.source_chunk_ids AS source_chunk_ids
    LIMIT $limit
    """
    rows = await tool._acypher(
        request,
        cypher,
        {"entity": entity, "entity_type": entity_type, "predicates": predicates, "limit": limit},
    )

    candidate_count = 0
    if rows:
        row0 = rows[0] if isinstance(rows[0], dict) else {}
        if isinstance(row0, dict) and "candidate_count" not in row0:
            candidate_count = 1
        else:
            raw_count = row0.get("candidate_count") if isinstance(row0, dict) else None
            try:
                candidate_count = int(raw_count) if raw_count is not None else 0
            except Exception:
                candidate_count = 0

    if candidate_count != 1:
        resolution_overrides = args.get("resolution") if isinstance(args.get("resolution"), dict) else {}
        resolver = build_default_entity_resolver(
            enabled=bool(resolution_overrides.get("enabled", NEIGHBORS_ENTITY_RESOLUTION_ENABLED)),
            candidate_limit=int(resolution_overrides.get("candidate_limit", NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT)),
            min_token_len=int(resolution_overrides.get("min_token_len", NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN)),
            min_token_hits=int(resolution_overrides.get("min_token_hits", NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS)),
            auto_score_min=float(resolution_overrides.get("auto_score_min", NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN)),
            auto_score_margin=float(resolution_overrides.get("auto_score_margin", NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN)),
        )
        resolution = await resolver.resolve(
            adapter=adapter,
            access_scope=request.access_scope,
            raw_entity=raw_entity,
            entity_type_hint=entity_type,
        )
        resolution_candidates = [_resolution_candidate_payload(c) for c in resolution.candidates]

        if resolution.resolved_candidate is None:
            summary = (
                "neighbors failed to match the provided entity name; returning similar entity candidates "
                "for disambiguation."
            )
            evidence = build_derived_evidence(
                tool_name=tool.descriptor.name,
                plan_step=request.plan_step,
                label="neighbors_resolution",
                content=f"neighbors_resolution: entity={entity} candidate_count={candidate_count}",
                provenance={
                    "entity": entity,
                    "entity_type": entity_type or None,
                    "candidate_count": candidate_count,
                    "resolution_candidates": resolution_candidates[: min(8, len(resolution_candidates))],
                    "resolution_diagnostics": dict(resolution.diagnostics),
                },
                kind=EVIDENCE_KIND_DIAGNOSTIC,
            )
            return ToolResult(
                summary=summary,
                evidences=[evidence],
                diagnostics={
                    "entity": entity,
                    "entity_type": entity_type or None,
                    "candidate_count": candidate_count,
                    "resolved": False,
                    "resolution_candidates": resolution_candidates,
                    "resolution_diagnostics": dict(resolution.diagnostics),
                },
            )

        resolved = resolution.resolved_candidate
        resolved_entity = {
            "entity_id": resolved.entity_id,
            "entity_name": resolved.entity_name,
            "entity_type": resolved.entity_type,
            "entity_type_key": resolved.entity_type_key,
            "score": resolved.score,
            "strategy": resolved.strategy,
            "edge_count": resolved.edge_count,
            "mention_count": resolved.mention_count,
        }
        neighbors = await _neighbors_by_entity_id(
            tool=tool,
            request=request,
            entity_id=resolved.entity_id,
            predicates=predicates,
            direction=direction,
            limit=limit,
        )
        summary = (
            f"neighbors: entity={entity} resolved={resolved.entity_name_normalized or resolved.entity_name} "
            f"direction={direction} count={len(neighbors)}"
        )
        evidence = build_derived_evidence(
            tool_name=tool.descriptor.name,
            plan_step=request.plan_step,
            label="neighbors",
            content=summary,
            provenance={
                "entity": entity,
                "entity_type": entity_type or None,
                "resolved_entity": resolved_entity,
                "resolution_candidates": resolution_candidates[: min(8, len(resolution_candidates))],
                "resolution_diagnostics": dict(resolution.diagnostics),
                "predicates": predicates,
                "direction": direction,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
                "neighbors": neighbors,
            },
            kind=EVIDENCE_KIND_DIAGNOSTIC,
        )
        return ToolResult(
            summary=summary,
            evidences=[evidence],
            diagnostics={
                "neighbors": neighbors,
                "resolved": True,
                "resolved_entity": resolved_entity,
                "resolution_candidates": resolution_candidates,
                "resolution_diagnostics": dict(resolution.diagnostics),
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
            },
        )

    neighbors: List[Dict[str, Any]] = []
    for row in rows or []:
        neighbor = str((row or {}).get("neighbor") or "").strip()
        predicate = str((row or {}).get("predicate") or "").strip()
        if not neighbor or not predicate:
            continue
        neighbors.append(
            {
                "neighbor": neighbor,
                "predicate": predicate,
                "fact_id": (row or {}).get("fact_id"),
                "source_chunk_ids": (row or {}).get("source_chunk_ids") or [],
            }
        )
    summary = f"neighbors: entity={entity} direction={direction} count={len(neighbors)}"
    content = summary
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="neighbors",
        content=content,
        provenance={
            "entity": entity,
            "entity_type": entity_type or None,
            "predicates": predicates,
            "direction": direction,
            "direction_forced_sensitive": forced_sensitive,
            "direction_forced_undirected": forced_undirected,
            "neighbors": neighbors,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(
        summary=summary,
        evidences=[evidence],
        diagnostics={"neighbors": neighbors, "resolved_entity": None, "resolved": False},
    )


async def _neighbors_by_entity_id(
    *,
    tool,
    request: ToolRunRequest,
    entity_id: str,
    predicates: List[str],
    direction: str,
    limit: int,
) -> List[Dict[str, Any]]:
    rel = rel_pattern(direction, rel_var="r", rel_type="RELATES_TO")
    predicate_clause = "AND r.predicate IN $predicates" if predicates else ""
    cypher = f"""
    MATCH (e:Entity)
    WHERE COALESCE(e.owner_id, $global_owner) = $owner_id
      AND e.entity_id = $entity_id
    MATCH (e){rel}(n:Entity)
    WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
      AND COALESCE(n.owner_id, $global_owner) = $owner_id
      {predicate_clause}
    RETURN n.entity_name AS neighbor,
           r.predicate AS predicate,
           r.fact_id AS fact_id,
           r.source_chunk_ids AS source_chunk_ids
    LIMIT $limit
    """
    rows = await tool._acypher(
        request,
        cypher,
        {"entity_id": entity_id, "predicates": predicates, "limit": limit},
    )
    neighbors: List[Dict[str, Any]] = []
    for row in rows or []:
        neighbor = str((row or {}).get("neighbor") or "").strip()
        predicate = str((row or {}).get("predicate") or "").strip()
        if not neighbor:
            continue
        neighbors.append(
            {
                "neighbor": neighbor,
                "predicate": predicate,
                "fact_id": (row or {}).get("fact_id"),
                "source_chunk_ids": (row or {}).get("source_chunk_ids") or [],
            }
        )
    return neighbors


async def run_trace_to_root(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops trace_to_root requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    raw_leaf = str(args.get("leaf") or "").strip()
    leaf = normalize_entity_name(raw_leaf)
    if not leaf:
        return ToolResult(summary="trace_to_root requires a non-empty leaf entity name.")

    leaf_type = str(args.get("leaf_type") or "").strip()
    predicates = normalize_predicates(args.get("predicates"))
    max_hops = limit_int(
        args.get("max_hops"),
        GRAPH_OPS_TRACE_TO_ROOT_DEFAULT_MAX_HOPS,
        max_value=GRAPH_OPS_TRACE_TO_ROOT_MAX_HOPS,
    )
    predicate_clause = "AND ALL(r IN relationships(p) WHERE r.predicate IN $predicates)" if predicates else ""
    incoming_root_clause = "AND r0.predicate IN $predicates" if predicates else ""

    cypher = f"""
    MATCH (leaf0:Entity)
    WHERE COALESCE(leaf0.owner_id, $global_owner) = $owner_id
      AND leaf0.entity_name_normalized = $leaf
      AND ($leaf_type = '' OR leaf0.entity_type = $leaf_type)
    WITH collect(leaf0) AS leaf_nodes
    WITH size(leaf_nodes) AS leaf_candidates,
         CASE WHEN size(leaf_nodes) = 1 THEN leaf_nodes[0] ELSE NULL END AS leaf
    MATCH p=(root:Entity)-[:RELATES_TO*1..{max_hops}]->(leaf)
    WHERE COALESCE(root.owner_id, $global_owner) = $owner_id
      AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
      {predicate_clause}
      AND NOT EXISTS {{
        MATCH (:Entity)-[r0:RELATES_TO]->(root)
        WHERE COALESCE(r0.owner_id, $global_owner) = $owner_id
          {incoming_root_clause}
      }}
    RETURN leaf_candidates AS leaf_candidates,
           [n IN nodes(p) | n.entity_name] AS chain,
           length(p) AS hops
    ORDER BY hops DESC
    LIMIT 1
    """
    rows = await tool._acypher(
        request,
        cypher,
        {"leaf": leaf, "predicates": predicates, "leaf_type": leaf_type},
    )

    row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
    leaf_candidates = int((row0 or {}).get("leaf_candidates") or 1)
    if leaf_candidates != 1:
        return ToolResult(
            summary=(
                "trace_to_root aborted due to ambiguous leaf entity name. "
                f"leaf_candidates={leaf_candidates}. Provide leaf_type to disambiguate."
            ),
            diagnostics={"leaf": leaf, "leaf_candidates": leaf_candidates, "leaf_type": leaf_type or None},
        )

    chain = (row0 or {}).get("chain") or []
    if not chain:
        return ToolResult(summary="trace_to_root query returned no chain.", diagnostics={"leaf": leaf, "predicates": predicates})

    chain = unique_strings([str(item) for item in chain])
    summary = f"trace_to_root: leaf={leaf} hops={len(chain) - 1}"
    content = "chain: " + " -> ".join(chain)
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="chain",
        content=content,
        provenance={"leaf": leaf, "predicates": predicates, "chain": chain},
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"chain": chain})
