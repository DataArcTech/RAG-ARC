"""Deterministic traversal tools backed by Neo4j Cypher."""
from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.entity_resolution import build_default_entity_resolver
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DIAGNOSTIC, EVIDENCE_KIND_DERIVED

from config.core.deepsearch.tool_defaults import (
    NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN,
    NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN,
    NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT,
    NEIGHBORS_ENTITY_RESOLUTION_ENABLED,
    NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS,
    NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN,
)
from core.graph_adapter.cypher import adapter_supports_cypher
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, REQUIRES_CYPHER, SCOPE_OWNER
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


def _resolution_candidate_payload(candidate: Any) -> Dict[str, Any]:
    """Expose resolver candidates in tool diagnostics without leaking internal objects."""

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


class GraphPathExistsTool(GraphTool):
    """Deterministic multi-hop path existence query."""

    descriptor = ToolDescriptor(
        name="graph.path_exists",
        channel="graph",
        description=(
            "Deterministic path-existence query (shortest path) backed by Neo4j Cypher. "
            "Evidence: citeable when `fact_ids/source_chunk_ids` are present in provenance."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("path", "traversal", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
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
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Path query requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        raw_source = str(request.extra.get("source") or "").strip()
        raw_target = str(request.extra.get("target") or "").strip()
        source = normalize_entity_name(raw_source)
        target = normalize_entity_name(raw_target)
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
            resolver = build_default_entity_resolver(
                enabled=True,
                candidate_limit=NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT,
                min_token_len=NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN,
                min_token_hits=NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS,
                auto_score_min=NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN,
                auto_score_margin=NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN,
            )
            src_res = await resolver.resolve(
                adapter=adapter,
                access_scope=request.access_scope,
                raw_entity=raw_source,
                entity_type_hint=source_type,
            )
            tgt_res = await resolver.resolve(
                adapter=adapter,
                access_scope=request.access_scope,
                raw_entity=raw_target,
                entity_type_hint=target_type,
            )

            if src_res.resolved_candidate is not None and tgt_res.resolved_candidate is not None:
                cypher_ids = f"""
                MATCH (s:Entity {{entity_id: $source_id}})
                WHERE COALESCE(s.owner_id, $global_owner) = $owner_id
                MATCH (t:Entity {{entity_id: $target_id}})
                WHERE COALESCE(t.owner_id, $global_owner) = $owner_id
                OPTIONAL MATCH p = (s){rel}(t)
                WHERE ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
                  {predicate_filter}
                RETURN CASE WHEN p IS NULL THEN [] ELSE [n IN nodes(p) | n.entity_name] END AS nodes,
                       CASE WHEN p IS NULL THEN [] ELSE [r IN relationships(p) | r.predicate] END AS predicates,
                       CASE WHEN p IS NULL THEN [] ELSE [r IN relationships(p) | r.fact_id] END AS fact_ids,
                       CASE WHEN p IS NULL THEN [] ELSE [r IN relationships(p) | r.source_chunk_ids] END AS source_chunk_ids
                ORDER BY COALESCE(length(p), 999) ASC
                LIMIT 1
                """
                async with adapter_locked(adapter):
                    rows2 = await adapter.acypher(
                        cypher_ids,
                        {
                            "source_id": src_res.resolved_candidate.entity_id,
                            "target_id": tgt_res.resolved_candidate.entity_id,
                            "predicates": predicates,
                        },
                        access_scope=request.access_scope,
                    )
                row2 = (rows2 or [{}])[0] if isinstance(rows2, list) else {}
                nodes2 = (row2 or {}).get("nodes") or []
                ok2 = bool(nodes2) and len(nodes2) >= 2
                summary2 = f"path_exists ok={ok2}: {source} -> {target} (max_hops={max_hops})"
                if not ok2:
                    return ToolResult(
                        summary=summary2,
                        diagnostics={
                            "ok": False,
                            "source": source,
                            "target": target,
                            "max_hops": max_hops,
                            "resolution": {
                                "source": dict(src_res.diagnostics),
                                "target": dict(tgt_res.diagnostics),
                            },
                        },
                    )

                content2 = "path: " + " -> ".join(str(n) for n in nodes2)
                chunk_id2 = derived_chunk_id(
                    tool_name=self.descriptor.name,
                    plan_step=request.plan_step,
                    label="path",
                    content=content2,
                )
                evidence2 = EvidenceChunk(
                    chunk_id=chunk_id2,
                    source=self.descriptor.name,
                    content=content2,
                    kind=EVIDENCE_KIND_DERIVED,
                    provenance={
                        "nodes": nodes2,
                        "predicates": (row2 or {}).get("predicates") or [],
                        "fact_ids": (row2 or {}).get("fact_ids") or [],
                        "source_chunk_ids": (row2 or {}).get("source_chunk_ids") or [],
                        "direction": direction,
                        "direction_forced_sensitive": forced_sensitive,
                        "direction_forced_undirected": forced_undirected,
                        "allowed_predicates": predicates,
                        "resolution": {
                            "source": dict(src_res.diagnostics),
                            "target": dict(tgt_res.diagnostics),
                        },
                    },
                )
                return ToolResult(
                    summary=summary2,
                    evidences=[evidence2],
                    diagnostics={
                        "ok": True,
                        "nodes": nodes2,
                        "resolution": {
                            "source": dict(src_res.diagnostics),
                            "target": dict(tgt_res.diagnostics),
                        },
                    },
                )

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
                    "resolution_candidates": {
                        "source": [_resolution_candidate_payload(c) for c in src_res.candidates],
                        "target": [_resolution_candidate_payload(c) for c in tgt_res.candidates],
                    },
                    "resolution_diagnostics": {
                        "source": dict(src_res.diagnostics),
                        "target": dict(tgt_res.diagnostics),
                    },
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
            kind=EVIDENCE_KIND_DERIVED,
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

    def __init__(
        self,
        *,
        enable_entity_resolution: bool = NEIGHBORS_ENTITY_RESOLUTION_ENABLED,
        resolution_candidate_limit: int = NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT,
        resolution_min_token_len: int = NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN,
        resolution_min_token_hits: int = NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS,
        auto_resolve_min_score: float = NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN,
        auto_resolve_score_margin: float = NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN,
    ) -> None:
        # Avoid hidden hard-coded thresholds: defaults live in config/core/deepsearch/tool_defaults.py.
        self._entity_resolver = build_default_entity_resolver(
            enabled=bool(enable_entity_resolution),
            candidate_limit=int(resolution_candidate_limit),
            min_token_len=int(resolution_min_token_len),
            min_token_hits=int(resolution_min_token_hits),
            auto_score_min=float(auto_resolve_min_score),
            auto_score_margin=float(auto_resolve_score_margin),
        )

    descriptor = ToolDescriptor(
        name="graph.neighbors",
        channel="graph",
        description=(
            "Deterministic 1-hop neighbor lookup backed by Neo4j Cypher. "
            "Evidence: citeable when `fact_id/source_chunk_ids` are present in provenance."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("neighbors", "direction", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
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
        example_args={
            "question": "What entities are 1-hop away from OpenAI via founded_by?",
            "plan_step": "plan_01",
            "extra": {"entity": "OpenAI", "predicates": ["FOUNDED_BY"], "direction": "out", "limit": 10},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Neighbors query requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        raw_entity = str(request.extra.get("entity") or "").strip()
        entity = normalize_entity_name(raw_entity)
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
        resolved_entity: Dict[str, Any] | None = None

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
        OPTIONAL MATCH (e){rel}(n:Entity)
        WHERE e IS NOT NULL
          AND COALESCE(r.owner_id, $global_owner) = $owner_id
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

        candidate_count = 0
        if rows:
            row0 = rows[0] if isinstance(rows[0], dict) else {}
            # Backward-compatible behavior for stubs/older adapters that return only neighbor rows.
            if isinstance(row0, dict) and "candidate_count" not in row0:
                candidate_count = 1
            else:
                raw_count = (row0 or {}).get("candidate_count") if isinstance(row0, dict) else None
                try:
                    candidate_count = int(raw_count) if raw_count is not None else 0
                except Exception:
                    candidate_count = 0
        if candidate_count != 1:
            resolution = await self._entity_resolver.resolve(
                adapter=adapter,
                access_scope=request.access_scope,
                raw_entity=raw_entity,
                entity_type_hint=entity_type,
            )
            resolution_candidates = [_resolution_candidate_payload(c) for c in resolution.candidates]

            if resolution.resolved_candidate is None:
                summary = (
                    "Neighbors query failed to match the provided entity name; "
                    "returning similar entity candidates for disambiguation."
                )
                chunk_id = derived_chunk_id(
                    tool_name=self.descriptor.name,
                    plan_step=request.plan_step,
                    label="neighbors_resolution",
                    content=f"neighbors_resolution: entity={entity} candidate_count={candidate_count}",
                )
                evidence = EvidenceChunk(
                    chunk_id=chunk_id,
                    source=self.descriptor.name,
                    content=f"neighbors_resolution: entity={entity} candidates={len(resolution_candidates)}",
                    kind=EVIDENCE_KIND_DIAGNOSTIC,
                    provenance={
                        "entity": entity,
                        "entity_type": entity_type or None,
                        "candidate_count": candidate_count,
                        "resolution_candidates": resolution_candidates[: min(8, len(resolution_candidates))],
                        "resolution_diagnostics": dict(resolution.diagnostics),
                    },
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
            neighbors = await self._neighbors_by_entity_id(
                adapter=adapter,
                entity_id=resolved.entity_id,
                predicates=predicates,
                direction=direction,
                limit=limit,
                access_scope=request.access_scope,
            )
            summary = (
                f"neighbors: entity={entity} resolved={resolved.entity_name_normalized or resolved.entity_name} "
                f"direction={direction} count={len(neighbors)}"
            )
            chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="neighbors", content=summary)
            evidence = EvidenceChunk(
                chunk_id=chunk_id,
                source=self.descriptor.name,
                content=summary,
                kind=EVIDENCE_KIND_DIAGNOSTIC,
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
            kind=EVIDENCE_KIND_DIAGNOSTIC,
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
                "resolved_entity": resolved_entity,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
            },
        )

    async def _neighbors_by_entity_id(
        self,
        *,
        adapter,
        entity_id: str,
        predicates: List[str],
        direction: str,
        limit: int,
        access_scope,
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
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"entity_id": str(entity_id), "predicates": predicates, "limit": int(limit)},
                access_scope=access_scope,
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
        description=(
            "Deterministic hierarchy tracing that returns a root-to-leaf chain backed by Neo4j Cypher. "
            "Evidence: citeable when the hierarchy edges have `source_chunk_ids`."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("hierarchy", "lineage", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
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
        example_args={
            "question": "Trace OpenAI department hierarchy to the root",
            "plan_step": "plan_06",
            "extra": {"leaf": "OpenAI Research Team", "predicates": ["CONTAINS"], "max_hops": 6},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Trace-to-root query requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        raw_leaf = str(request.extra.get("leaf") or "").strip()
        leaf = normalize_entity_name(raw_leaf)
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
            resolver = build_default_entity_resolver(
                enabled=True,
                candidate_limit=NEIGHBORS_ENTITY_RESOLUTION_CANDIDATE_LIMIT,
                min_token_len=NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_LEN,
                min_token_hits=NEIGHBORS_ENTITY_RESOLUTION_MIN_TOKEN_HITS,
                auto_score_min=NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MIN,
                auto_score_margin=NEIGHBORS_ENTITY_RESOLUTION_AUTO_SCORE_MARGIN,
            )
            leaf_res = await resolver.resolve(
                adapter=adapter,
                access_scope=request.access_scope,
                raw_entity=raw_leaf,
                entity_type_hint=leaf_type,
            )
            if leaf_res.resolved_candidate is not None:
                cypher2 = f"""
                MATCH (leaf:Entity {{entity_id: $leaf_id}})
                WHERE COALESCE(leaf.owner_id, $global_owner) = $owner_id
                MATCH p=(root:Entity)-[:RELATES_TO*1..{max_hops}]->(leaf)
                WHERE COALESCE(root.owner_id, $global_owner) = $owner_id
                  AND ALL(r IN relationships(p) WHERE COALESCE(r.owner_id, $global_owner) = $owner_id)
                  {predicate_clause}
                  AND NOT EXISTS {{
                    MATCH (:Entity)-[r0:RELATES_TO]->(root)
                    WHERE COALESCE(r0.owner_id, $global_owner) = $owner_id
                      {incoming_root_clause}
                  }}
                RETURN [n IN nodes(p) | n.entity_name] AS chain,
                       length(p) AS hops
                ORDER BY hops DESC
                LIMIT 1
                """
                async with adapter_locked(adapter):
                    rows2 = await adapter.acypher(
                        cypher2,
                        {"leaf_id": leaf_res.resolved_candidate.entity_id, "predicates": predicates},
                        access_scope=request.access_scope,
                    )
                row2 = (rows2 or [{}])[0] if isinstance(rows2, list) else {}
                chain2 = (row2 or {}).get("chain") or []
                if chain2:
                    chain2 = [str(item) for item in chain2 if str(item).strip()]
                    summary2 = f"trace_to_root: leaf={leaf} hops={len(chain2) - 1}"
                    content2 = "chain: " + " -> ".join(chain2)
                    chunk_id2 = derived_chunk_id(
                        tool_name=self.descriptor.name,
                        plan_step=request.plan_step,
                        label="chain",
                        content=content2,
                    )
                    evidence2 = EvidenceChunk(
                        chunk_id=chunk_id2,
                        source=self.descriptor.name,
                        content=content2,
                        kind=EVIDENCE_KIND_DERIVED,
                        provenance={
                            "leaf": leaf,
                            "predicates": predicates,
                            "chain": chain2,
                            "resolution": dict(leaf_res.diagnostics),
                        },
                    )
                    return ToolResult(
                        summary=summary2,
                        evidences=[evidence2],
                        diagnostics={"chain": chain2, "resolution": dict(leaf_res.diagnostics)},
                    )
            return ToolResult(
                summary=(
                    "Trace-to-root aborted due to ambiguous leaf entity name. "
                    f"leaf_candidates={leaf_candidates}. Provide leaf_type to disambiguate."
                ),
                diagnostics={
                    "leaf": leaf,
                    "leaf_candidates": leaf_candidates,
                    "leaf_type": leaf_type or None,
                    "resolution_candidates": [_resolution_candidate_payload(c) for c in leaf_res.candidates],
                    "resolution_diagnostics": dict(leaf_res.diagnostics),
                },
            )
        chain = (row0 or {}).get("chain") or []
        if not chain:
            return ToolResult(summary="Trace-to-root query returned no chain.", diagnostics={"leaf": leaf, "predicates": predicates})

        chain = [str(item) for item in chain if str(item).strip()]
        summary = f"trace_to_root: leaf={leaf} hops={len(chain) - 1}"
        content = "chain: " + " -> ".join(chain)
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="chain", content=content)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=content,
            kind=EVIDENCE_KIND_DERIVED,
            provenance={"leaf": leaf, "predicates": predicates, "chain": chain},
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"chain": chain})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphTraceToRootTool requires a GraphDeepSearchAdapter instance")
        return adapter
