"""Deterministic traversal tools backed by Neo4j Cypher."""
from typing import Any, Dict, List

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
    rel_pattern_varlen,
)


class GraphPathExistsTool(GraphTool):
    """Deterministic multi-hop path existence query."""

    descriptor = ToolDescriptor(
        name="graph.path_exists",
        channel="graph",
        description="Deterministic path-existence query (shortest path) backed by Neo4j Cypher.",
        speed="fast",
        cost="low",
        strategy_tags=("path", "traversal", "deterministic"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_path_exists",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "source": {"type": "string", "description": "Source entity name."},
                "source_type": {"type": "string", "description": "Optional source entity_type for disambiguation."},
                "target": {"type": "string", "description": "Target entity name."},
                "target_type": {"type": "string", "description": "Optional target entity_type for disambiguation."},
                "predicates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional allowed predicates along the path.",
                },
                "direction": {"type": "string", "enum": ["out", "in", "both"], "description": "Traversal direction."},
                "max_hops": {"type": "integer", "minimum": 1, "description": "Max hop count for path search."},
            },
            required_extra_fields=("source", "target"),
        ),
        example_args={
            "question": "Does A company transitively own C company?",
            "plan_step": "plan_01",
            "extra": {"source": "A公司", "target": "C公司", "predicates": ["OWNS"], "direction": "out", "max_hops": 5},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not isinstance(adapter, GraphCypherQueryable) or not adapter.cypher_capable():
            return ToolResult(
                summary="Path query requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        source = normalize_entity_name(request.extra.get("source"))
        target = normalize_entity_name(request.extra.get("target"))
        if not source or not target:
            return ToolResult(summary="Path query requires non-empty source/target entity names.")

        predicates = normalize_predicates(request.extra.get("predicates"))
        source_type = str(request.extra.get("source_type") or "").strip()
        target_type = str(request.extra.get("target_type") or "").strip()
        direction_raw = str(request.extra.get("direction") or "out")
        directionality = directionality_config(adapter)
        direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
            direction_raw, predicates, directionality=directionality, default_direction="out"
        )
        direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            direction, predicates, directionality=directionality
        )
        max_hops = limit_int(request.extra.get("max_hops"), 4, max_value=20)

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
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {
                    "source": source,
                    "target": target,
                    "predicates": predicates,
                    "source_type": source_type,
                    "target_type": target_type,
                },
                access_scope=request.access_scope,
            )

        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        source_candidates = int((row0 or {}).get("source_candidates") or 1)
        target_candidates = int((row0 or {}).get("target_candidates") or 1)
        if source_candidates != 1 or target_candidates != 1:
            return ToolResult(
                summary=(
                    "Path query aborted due to ambiguous entities. "
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
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="path", content=content)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
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
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"ok": True, "nodes": nodes})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphPathExistsTool requires a GraphDeepSearchAdapter instance")
        return adapter


class GraphNeighborsTool(GraphTool):
    """Deterministic 1-hop neighbor query (successors/predecessors)."""

    descriptor = ToolDescriptor(
        name="graph.neighbors",
        channel="graph",
        description="Deterministic 1-hop neighbor lookup backed by Neo4j Cypher.",
        speed="fast",
        cost="low",
        strategy_tags=("neighbors", "direction", "deterministic"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_neighbors",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "entity": {"type": "string", "description": "Anchor entity name."},
                "entity_type": {"type": "string", "description": "Optional entity_type for disambiguation."},
                "predicates": {"type": "array", "items": {"type": "string"}, "description": "Optional predicate filters."},
                "direction": {"type": "string", "enum": ["out", "in", "both"], "description": "Direction for traversal."},
                "limit": {"type": "integer", "minimum": 1, "description": "Max neighbors returned."},
            },
            required_extra_fields=("entity",),
        ),
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not isinstance(adapter, GraphCypherQueryable) or not adapter.cypher_capable():
            return ToolResult(
                summary="Neighbors query requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        entity = normalize_entity_name(request.extra.get("entity"))
        if not entity:
            return ToolResult(summary="Neighbors requires a non-empty entity name.")
        predicates = normalize_predicates(request.extra.get("predicates"))
        entity_type = str(request.extra.get("entity_type") or "").strip()
        direction_raw = str(request.extra.get("direction") or "out")
        directionality = directionality_config(adapter)
        direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
            direction_raw, predicates, directionality=directionality, default_direction="out"
        )
        direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            direction, predicates, directionality=directionality
        )
        limit = limit_int(request.extra.get("limit"), 25, max_value=200)

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
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"entity": entity, "predicates": predicates, "limit": limit, "entity_type": entity_type},
                access_scope=request.access_scope,
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
                    "Neighbors query aborted due to ambiguous entity name. "
                    f"candidate_count={candidate_count}. Provide entity_type to disambiguate."
                ),
                diagnostics={"entity": entity, "candidate_count": candidate_count, "entity_type": entity_type or None},
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
        summary = f"neighbors: entity={entity} direction={direction} count={len(neighbors)}"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="neighbors", content=summary)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=summary,
            provenance={
                "entity": entity,
                "predicates": predicates,
                "direction": direction,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
                "neighbors": neighbors,
            },
        )
        return ToolResult(
            summary=summary,
            evidences=[evidence],
            diagnostics={
                "neighbors": neighbors,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
            },
        )

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphNeighborsTool requires a GraphDeepSearchAdapter instance")
        return adapter


class GraphTraceToRootTool(GraphTool):
    """Deterministic lineage tracing (leaf -> root) assuming a hierarchical predicate."""

    descriptor = ToolDescriptor(
        name="graph.trace_to_root",
        channel="graph",
        description="Deterministic hierarchy tracing that returns a root-to-leaf chain backed by Neo4j Cypher.",
        speed="fast",
        cost="low",
        strategy_tags=("hierarchy", "lineage", "deterministic"),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_trace_to_root",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "leaf": {"type": "string", "description": "Leaf entity name (starting point)."},
                "leaf_type": {"type": "string", "description": "Optional leaf entity_type for disambiguation."},
                "predicates": {"type": "array", "items": {"type": "string"}, "description": "Hierarchy predicates (e.g., CONTAINS)."},
                "max_hops": {"type": "integer", "minimum": 1, "description": "Max hop count for tracing."},
            },
            required_extra_fields=("leaf",),
        ),
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not isinstance(adapter, GraphCypherQueryable) or not adapter.cypher_capable():
            return ToolResult(
                summary="Trace-to-root query requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        leaf = normalize_entity_name(request.extra.get("leaf"))
        if not leaf:
            return ToolResult(summary="Trace-to-root requires a non-empty leaf entity name.")

        leaf_type = str(request.extra.get("leaf_type") or "").strip()
        predicates = normalize_predicates(request.extra.get("predicates"))
        max_hops = limit_int(request.extra.get("max_hops"), 6, max_value=20)
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

        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"leaf": leaf, "predicates": predicates, "leaf_type": leaf_type},
                access_scope=request.access_scope,
            )

        row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
        leaf_candidates = int((row0 or {}).get("leaf_candidates") or 1)
        if leaf_candidates != 1:
            return ToolResult(
                summary=(
                    "Trace-to-root aborted due to ambiguous leaf entity name. "
                    f"leaf_candidates={leaf_candidates}. Provide leaf_type to disambiguate."
                ),
                diagnostics={"leaf": leaf, "leaf_candidates": leaf_candidates, "leaf_type": leaf_type or None},
            )
        chain = (row0 or {}).get("chain") or []
        if not chain:
            return ToolResult(summary="Trace-to-root query returned no chain.", diagnostics={"leaf": leaf, "predicates": predicates})

        chain = [str(item) for item in chain if str(item).strip()]
        summary = f"trace_to_root: leaf={leaf} hops={len(chain) - 1}"
        content = "chain: " + " -> ".join(chain)
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="chain", content=content)
        evidence = EvidenceChunk(chunk_id=chunk_id, source=self.descriptor.name, content=content, provenance={"leaf": leaf, "predicates": predicates, "chain": chain})
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"chain": chain})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphTraceToRootTool requires a GraphDeepSearchAdapter instance")
        return adapter
