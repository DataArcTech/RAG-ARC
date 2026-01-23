"""Templates for fact lookups, schema nodes, and concept expansion."""
from typing import Any, Dict, List

from core.deepsearch.tools.base import ToolResult, ToolRunRequest
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.graph_adapter.cypher import adapter_supports_cypher

from config.core.deepsearch.tool_defaults import (
    GRAPH_OPS_ENTITY_CONCEPTS_DEFAULT_LIMIT,
    GRAPH_OPS_ENTITY_CONCEPTS_MAX_LIMIT,
    GRAPH_OPS_EXPAND_TERMS_DEFAULT_LIMIT,
    GRAPH_OPS_EXPAND_TERMS_MAX_LIMIT,
    GRAPH_OPS_FACTS_BY_TYPE_DEFAULT_LIMIT,
    GRAPH_OPS_FACTS_BY_TYPE_MAX_LIMIT,
    GRAPH_OPS_SCHEMA_NODES_DEFAULT_LIMIT,
    GRAPH_OPS_SCHEMA_NODES_MAX_LIMIT,
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
from .templates_utils import build_derived_evidence, unique_strings


async def run_entity_concepts(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops entity_concepts requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    entity = normalize_entity_name(args.get("entity"))
    entity_type = str(args.get("entity_type") or "").strip()
    term = normalize_entity_name(args.get("term"))
    limit = limit_int(
        args.get("limit"),
        GRAPH_OPS_ENTITY_CONCEPTS_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_ENTITY_CONCEPTS_MAX_LIMIT,
    )

    if not entity and not term:
        return ToolResult(summary="entity_concepts requires entity or term.")

    if entity:
        cypher = """
        MATCH (e0:Entity)
        WHERE COALESCE(e0.owner_id, $global_owner) = $owner_id
          AND e0.entity_name_normalized = $entity
          AND ($entity_type = '' OR e0.entity_type = $entity_type)
        WITH collect(e0) AS candidates
        WITH size(candidates) AS candidate_count,
             CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS e
        OPTIONAL MATCH (e)-[:CANONICAL_OF]->(c:EntityCanonical)
        WHERE COALESCE(c.owner_id, $global_owner) = $owner_id
        OPTIONAL MATCH (a:EntityAlias)-[:ALIAS_OF]->(c)
        WHERE COALESCE(a.owner_id, $global_owner) = $owner_id
        RETURN candidate_count AS candidate_count,
               e.entity_name AS entity_name,
               e.entity_type AS entity_type,
               c.canonical_id AS canonical_id,
               c.canonical_key AS canonical_key,
               c.canonical_name AS canonical_name,
               collect(DISTINCT a.alias_text)[..$limit] AS aliases
        LIMIT 1
        """
        params = {"entity": entity, "entity_type": entity_type, "limit": limit}
    else:
        cypher = """
        MATCH (a:EntityAlias)
        WHERE COALESCE(a.owner_id, $global_owner) = $owner_id
          AND a.alias_text_normalized CONTAINS $term
        MATCH (a)-[:ALIAS_OF]->(c:EntityCanonical)
        WHERE COALESCE(c.owner_id, $global_owner) = $owner_id
        WITH c, collect(DISTINCT a.alias_text)[..$limit] AS aliases
        RETURN c.canonical_id AS canonical_id,
               c.canonical_key AS canonical_key,
               c.canonical_name AS canonical_name,
               aliases AS aliases
        ORDER BY size(aliases) DESC, c.canonical_name ASC
        LIMIT $limit
        """
        params = {"term": term, "limit": limit}

    rows = await tool._acypher(request, cypher, params)

    if entity:
        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        candidate_count = int((row0 or {}).get("candidate_count") or 0)
        if candidate_count != 1:
            return ToolResult(
                summary=(
                    "entity_concepts aborted due to ambiguous or missing entity name. "
                    f"candidate_count={candidate_count}. Provide entity_type to disambiguate."
                ),
                diagnostics={"entity": entity, "candidate_count": candidate_count, "entity_type": entity_type or None},
            )
        concept = {
            "entity_name": (row0 or {}).get("entity_name"),
            "entity_type": (row0 or {}).get("entity_type"),
            "canonical_id": (row0 or {}).get("canonical_id"),
            "canonical_key": (row0 or {}).get("canonical_key"),
            "canonical_name": (row0 or {}).get("canonical_name"),
            "aliases": list((row0 or {}).get("aliases") or []),
        }
        summary = f"entity_concepts: entity={entity} canonical={concept.get('canonical_name') or 'unknown'} aliases={len(concept['aliases'])}"
        provenance: Dict[str, Any] = {"entity": entity, "entity_type": entity_type or None, "result": concept}
        label = "entity"
    else:
        results: List[Dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            canonical_name = str(row.get("canonical_name") or "").strip()
            if not canonical_name:
                continue
            results.append(
                {
                    "canonical_id": row.get("canonical_id"),
                    "canonical_key": row.get("canonical_key"),
                    "canonical_name": canonical_name,
                    "aliases": row.get("aliases") or [],
                }
            )
        summary = f"entity_concepts: term={term} matches={len(results)}"
        provenance = {"term": term or None, "results": results}
        label = "term"

    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label=label,
        content=summary,
        provenance=provenance,
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics=provenance)


async def run_schema_nodes(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops schema_nodes requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    chunk_id = str(args.get("chunk_id") or "").strip()
    term = str(args.get("term") or "").strip()
    layer = str(args.get("layer") or "").strip().lower()
    if layer not in {"concept", "process", "instance", "unknown", ""}:
        layer = ""
    limit = limit_int(
        args.get("limit"),
        GRAPH_OPS_SCHEMA_NODES_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_SCHEMA_NODES_MAX_LIMIT,
    )

    if not chunk_id and not term:
        return ToolResult(summary="schema_nodes requires chunk_id or term.")

    layer_clause = "AND s.layer = $layer" if layer else ""

    if chunk_id:
        cypher = f"""
        MATCH (c:Chunk {{chunk_id: $chunk_id}})
        WHERE COALESCE(c.owner_id, $global_owner) = $owner_id
        MATCH (c)-[r:HAS_SCHEMA_NODE]->(s:SchemaNode)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
          AND COALESCE(s.owner_id, $global_owner) = $owner_id
          {layer_clause}
        RETURN s.schema_id AS schema_id,
               s.layer AS layer,
               s.text AS text,
               s.text_normalized AS text_normalized,
               r.level AS level
        ORDER BY r.level ASC, s.text_normalized ASC
        LIMIT $limit
        """
        params = {"chunk_id": chunk_id, "layer": layer, "limit": limit}
    else:
        cypher = f"""
        MATCH (s:SchemaNode)
        WHERE COALESCE(s.owner_id, $global_owner) = $owner_id
          AND s.text_normalized CONTAINS $term
          {layer_clause}
        RETURN s.schema_id AS schema_id,
               s.layer AS layer,
               s.text AS text,
               s.text_normalized AS text_normalized,
               NULL AS level
        ORDER BY s.text_normalized ASC
        LIMIT $limit
        """
        params = {"term": term.lower(), "layer": layer, "limit": limit}

    rows = await tool._acypher(request, cypher, params)
    nodes: List[Dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        nodes.append(
            {
                "schema_id": row.get("schema_id"),
                "layer": row.get("layer"),
                "text": text,
                "text_normalized": row.get("text_normalized"),
                "level": row.get("level"),
            }
        )

    summary = f"schema_nodes: count={len(nodes)}"
    label = "chunk" if chunk_id else "term"
    provenance = {"chunk_id": chunk_id or None, "term": term or None, "layer": layer or None, "nodes": nodes}
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label=label,
        content=summary,
        provenance=provenance,
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"count": len(nodes)})


async def run_facts_by_type(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops facts_by_type requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    entity_type = str(args.get("entity_type") or "").strip()
    if not entity_type:
        return ToolResult(summary="facts_by_type requires a non-empty entity_type.")

    predicates = normalize_predicates(args.get("predicates"))
    limit = limit_int(
        args.get("limit"),
        GRAPH_OPS_FACTS_BY_TYPE_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_FACTS_BY_TYPE_MAX_LIMIT,
    )
    predicate_clause = "AND r.predicate IN $predicates" if predicates else ""
    direction_raw = str(args.get("direction") or "out")
    directionality = directionality_config(adapter)
    direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
        direction_raw, predicates, directionality=directionality, default_direction="out"
    )
    direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
        direction, predicates, directionality=directionality
    )

    if direction == "in":
        cypher = f"""
        MATCH (e:Entity)
        WHERE COALESCE(e.owner_id, $global_owner) = $owner_id
          AND e.entity_type = $entity_type
        MATCH (t:Entity)-[r:RELATES_TO]->(e)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
          AND COALESCE(t.owner_id, $global_owner) = $owner_id
          {predicate_clause}
        RETURN t.entity_name AS head,
               r.predicate AS predicate,
               e.entity_name AS tail,
               r.fact_id AS fact_id,
               r.source_chunk_ids AS source_chunk_ids
        LIMIT $limit
        """
    elif direction == "both":
        cypher = f"""
        MATCH (e:Entity)
        WHERE COALESCE(e.owner_id, $global_owner) = $owner_id
          AND e.entity_type = $entity_type
        MATCH (e)-[r:RELATES_TO]-(t:Entity)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
          AND COALESCE(t.owner_id, $global_owner) = $owner_id
          {predicate_clause}
        WITH startNode(r) AS s, endNode(r) AS tt, r AS r
        RETURN s.entity_name AS head,
               r.predicate AS predicate,
               tt.entity_name AS tail,
               r.fact_id AS fact_id,
               r.source_chunk_ids AS source_chunk_ids
        LIMIT $limit
        """
    else:
        cypher = f"""
        MATCH (e:Entity)
        WHERE COALESCE(e.owner_id, $global_owner) = $owner_id
          AND e.entity_type = $entity_type
        MATCH (e)-[r:RELATES_TO]->(t:Entity)
        WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
          AND COALESCE(t.owner_id, $global_owner) = $owner_id
          {predicate_clause}
        RETURN e.entity_name AS head,
               r.predicate AS predicate,
               t.entity_name AS tail,
               r.fact_id AS fact_id,
               r.source_chunk_ids AS source_chunk_ids
        LIMIT $limit
        """

    rows = await tool._acypher(
        request,
        cypher,
        {"entity_type": entity_type, "predicates": predicates, "limit": limit},
    )

    facts: List[Dict[str, Any]] = []
    for row in rows or []:
        head = str((row or {}).get("head") or "").strip()
        tail = str((row or {}).get("tail") or "").strip()
        predicate = str((row or {}).get("predicate") or "").strip()
        if not head or not tail or not predicate:
            continue
        facts.append(
            {
                "head": head,
                "predicate": predicate,
                "tail": tail,
                "fact_id": (row or {}).get("fact_id"),
                "source_chunk_ids": (row or {}).get("source_chunk_ids") or [],
            }
        )
    summary = f"facts_by_type: type={entity_type} direction={direction} count={len(facts)}"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="facts",
        content=summary,
        provenance={
            "entity_type": entity_type,
            "predicates": predicates,
            "direction": direction,
            "direction_forced_sensitive": forced_sensitive,
            "direction_forced_undirected": forced_undirected,
            "facts": facts,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"facts": facts})


async def run_expand_terms(tool, request: ToolRunRequest, args: Dict[str, Any]) -> ToolResult:
    adapter = request.adapter
    if adapter is None or not adapter_supports_cypher(adapter):
        return ToolResult(
            summary="graph.ops expand_terms requires a Cypher-capable graph adapter (Neo4j).",
            diagnostics={"reason": "cypher_unavailable"},
        )

    concept = normalize_entity_name(args.get("concept"))
    if not concept:
        return ToolResult(summary="expand_terms requires a non-empty concept.")

    concept_type = str(args.get("concept_type") or "").strip()
    predicates = normalize_predicates(args.get("predicates"))
    direction_raw = str(args.get("direction") or "in")
    directionality = directionality_config(adapter)
    direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
        direction_raw, predicates, directionality=directionality, default_direction="in"
    )
    direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
        direction, predicates, directionality=directionality
    )
    limit = limit_int(
        args.get("limit"),
        GRAPH_OPS_EXPAND_TERMS_DEFAULT_LIMIT,
        max_value=GRAPH_OPS_EXPAND_TERMS_MAX_LIMIT,
    )

    rel = rel_pattern(direction, rel_var="r", rel_type="RELATES_TO")
    predicate_clause = "AND r.predicate IN $predicates" if predicates else ""
    cypher = f"""
    MATCH (c0:Entity)
    WHERE COALESCE(c0.owner_id, $global_owner) = $owner_id
      AND c0.entity_name_normalized = $concept
      AND ($concept_type = '' OR c0.entity_type = $concept_type)
    WITH collect(c0) AS candidates
    WITH size(candidates) AS candidate_count,
         CASE WHEN size(candidates) = 1 THEN candidates[0] ELSE NULL END AS c
    MATCH (c){rel}(t:Entity)
    WHERE COALESCE(r.owner_id, $global_owner) = $owner_id
      AND COALESCE(t.owner_id, $global_owner) = $owner_id
      {predicate_clause}
    RETURN candidate_count AS candidate_count,
           t.entity_name AS term
    LIMIT $limit
    """
    rows = await tool._acypher(
        request,
        cypher,
        {"concept": concept, "predicates": predicates, "limit": limit, "concept_type": concept_type},
    )

    candidate_count = 1
    if rows:
        try:
            candidate_count = int((rows[0] or {}).get("candidate_count") or 1)
        except Exception:
            candidate_count = 1
    if candidate_count != 1:
        return ToolResult(
            summary=(
                "expand_terms aborted due to ambiguous concept entity name. "
                f"candidate_count={candidate_count}. Provide concept_type to disambiguate."
            ),
            diagnostics={"concept": concept, "candidate_count": candidate_count, "concept_type": concept_type or None},
        )

    terms = unique_strings(
        [str((row or {}).get("term") or "").strip() for row in rows or [] if str((row or {}).get("term") or "").strip()]
    )
    summary = f"expand_terms: concept={concept} expanded={len(terms)}"
    evidence = build_derived_evidence(
        tool_name=tool.descriptor.name,
        plan_step=request.plan_step,
        label="expanded",
        content=summary,
        provenance={
            "concept": concept,
            "predicates": predicates,
            "direction": direction,
            "direction_forced_sensitive": forced_sensitive,
            "direction_forced_undirected": forced_undirected,
            "terms": terms,
        },
        kind=EVIDENCE_KIND_DERIVED,
    )
    return ToolResult(summary=summary, evidences=[evidence], diagnostics={"terms": terms})
