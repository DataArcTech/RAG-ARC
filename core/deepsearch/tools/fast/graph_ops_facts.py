"""Deterministic fact lookup & term expansion tools backed by Neo4j Cypher."""
from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED

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
)


class GraphFactsByTypeTool(GraphTool):
    """Deterministic fact lookup filtered by entity type (disambiguation routing)."""

    descriptor = ToolDescriptor(
        name="graph.facts_by_type",
        channel="graph",
        description=(
            "Deterministic fact lookup for entities of a given type backed by Neo4j Cypher. "
            "Evidence: citeable when fact edges have `fact_id/source_chunk_ids`."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("type_filter", "disambiguation", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_facts_by_type",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "entity_type": {"type": "string", "description": "Entity type to filter on (e.g., Company)."},
                "predicates": {"type": "array", "items": {"type": "string"}, "description": "Optional predicate filters."},
                "direction": {
                    "type": "string",
                    "enum": ["out", "in", "both"],
                    "description": "Relationship direction to match (out|in|both) using Neo4j relationship direction.",
                },
                "limit": {"type": "integer", "minimum": 1, "description": "Max facts returned."},
            },
            required_extra_fields=("entity_type",),
        ),
        example_args={
            "question": "List facts for Company entities",
            "plan_step": "plan_02",
            "extra": {"entity_type": "Company", "direction": "out", "limit": 20},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Facts-by-type requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        entity_type = str(request.extra.get("entity_type") or "").strip()
        if not entity_type:
            return ToolResult(summary="Facts-by-type requires a non-empty entity_type.")

        predicates = normalize_predicates(request.extra.get("predicates"))
        limit = limit_int(request.extra.get("limit"), 50, max_value=300)
        predicate_clause = "AND r.predicate IN $predicates" if predicates else ""
        direction_raw = str(request.extra.get("direction") or "out")
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
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"entity_type": entity_type, "predicates": predicates, "limit": limit},
                access_scope=request.access_scope,
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
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="facts", content=summary)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=summary,
            kind=EVIDENCE_KIND_DERIVED,
            provenance={
                "entity_type": entity_type,
                "predicates": predicates,
                "direction": direction,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
                "facts": facts,
            },
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"facts": facts})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphFactsByTypeTool requires a GraphDeepSearchAdapter instance")
        return adapter


class GraphExpandTermsTool(GraphTool):
    """Deterministic ontology mapping / query expansion via graph edges."""

    descriptor = ToolDescriptor(
        name="graph.expand_terms",
        channel="graph",
        description=(
            "Deterministic query expansion: returns related terms via predicates (Cypher-backed). "
            "Evidence: citeable when the expansion edges have `source_chunk_ids`."
        ),
        speed="fast",
        cost="low",
        strategy_tags=("query_expansion", "ontology", "deterministic", EVIDENCE_PRIMARY, SCOPE_OWNER, REQUIRES_CYPHER),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_expand_terms",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "concept": {"type": "string", "description": "Anchor concept/entity name."},
                "concept_type": {"type": "string", "description": "Optional concept entity_type for disambiguation."},
                "predicates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Predicates used to expand from concept to related terms.",
                },
                "direction": {"type": "string", "enum": ["out", "in", "both"], "description": "Direction for expansion edges."},
                "limit": {"type": "integer", "minimum": 1, "description": "Max expanded terms returned."},
            },
            required_extra_fields=("concept",),
        ),
        example_args={
            "question": "Expand terms related to HippoRAG",
            "plan_step": "plan_01",
            "extra": {"concept": "HippoRAG", "direction": "both", "limit": 20},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Expand terms requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        concept = normalize_entity_name(request.extra.get("concept"))
        if not concept:
            return ToolResult(summary="Expand terms requires a non-empty concept.")

        concept_type = str(request.extra.get("concept_type") or "").strip()
        predicates = normalize_predicates(request.extra.get("predicates"))
        direction_raw = str(request.extra.get("direction") or "in")
        directionality = directionality_config(adapter)
        direction, forced_sensitive = enforce_direction_for_sensitive_predicates(
            direction_raw, predicates, directionality=directionality, default_direction="in"
        )
        direction, forced_undirected = enforce_undirected_for_non_sensitive_predicates(
            direction, predicates, directionality=directionality
        )
        limit = limit_int(request.extra.get("limit"), 25, max_value=200)

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
        async with adapter_locked(adapter):
            rows = await adapter.acypher(
                cypher,
                {"concept": concept, "predicates": predicates, "limit": limit, "concept_type": concept_type},
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
                    "Expand terms aborted due to ambiguous concept entity name. "
                    f"candidate_count={candidate_count}. Provide concept_type to disambiguate."
                ),
                diagnostics={"concept": concept, "candidate_count": candidate_count, "concept_type": concept_type or None},
            )

        terms = [str((row or {}).get("term") or "").strip() for row in rows or [] if str((row or {}).get("term") or "").strip()]
        terms = sorted(set(terms))
        summary = f"expand_terms: concept={concept} expanded={len(terms)}"
        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label="expanded", content=summary)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=summary,
            kind=EVIDENCE_KIND_DERIVED,
            provenance={
                "concept": concept,
                "predicates": predicates,
                "direction": direction,
                "direction_forced_sensitive": forced_sensitive,
                "direction_forced_undirected": forced_undirected,
                "terms": terms,
            },
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics={"terms": terms})

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphExpandTermsTool requires a GraphDeepSearchAdapter instance")
        return adapter
