"""Deterministic concept lookup over the canonicalization layer (Neo4j Cypher).

This tool makes the entity canonicalization layer queryable inside DeepSearch:
- Entity -> canonical concept (EntityCanonical)
- Canonical concept -> known aliases (EntityAlias)

This is a lightweight "concept layer" aligned with the repository's KG governance:
canonicalization is stored in-graph (no external fact store) and is deterministic.
"""

from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED

from core.graph_adapter.cypher import adapter_supports_cypher
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, build_input_schema
from ..governance_tags import EVIDENCE_PRIMARY, REQUIRES_CYPHER, SCOPE_OWNER
from .graph_ops_common import limit_int, normalize_entity_name


class GraphEntityConceptsTool(GraphTool):
    """Resolve entity/alias strings to canonical concept nodes and aliases."""

    descriptor = ToolDescriptor(
        name="graph.entity_concepts",
        channel="graph",
        description=(
            "Deterministic concept/canonical lookup over EntityCanonical/EntityAlias (Neo4j Cypher). "
            "Use for disambiguation + query expansion; citeable via provenance when available."
        ),
        speed="fast",
        cost="low",
        strategy_tags=(
            "concept",
            "canonicalization",
            "alias",
            "disambiguation",
            "deterministic",
            EVIDENCE_PRIMARY,
            SCOPE_OWNER,
            REQUIRES_CYPHER,
        ),
        profile="F",
        determinism="deterministic",
        namespace="rag-arc.deepsearch.tools.fast.graph_entity_concepts",
        mcp_callable=True,
        input_schema=build_input_schema(
            extra_properties={
                "entity": {"type": "string", "description": "Entity name to resolve (optional)."},
                "entity_type": {"type": "string", "description": "Optional entity_type to disambiguate Entity matches."},
                "term": {"type": "string", "description": "Alias search term (optional). Searches EntityAlias.alias_text_normalized."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Max results/aliases returned."},
            },
            required_extra_fields=(),
        ),
        example_args={
            "question": "Resolve entity aliases for disambiguation",
            "plan_step": "plan_02",
            "extra": {"term": "捷豹", "limit": 10},
        },
    )

    async def run(self, request: ToolRunRequest) -> ToolResult:
        adapter = self._require_adapter(request.adapter)
        if not adapter_supports_cypher(adapter):
            return ToolResult(
                summary="Entity concept lookup requires a Cypher-capable graph adapter (Neo4j).",
                diagnostics={"reason": "cypher_unavailable"},
            )

        entity = normalize_entity_name(request.extra.get("entity"))
        entity_type = str(request.extra.get("entity_type") or "").strip()
        term = normalize_entity_name(request.extra.get("term"))
        limit = limit_int(request.extra.get("limit"), 50, max_value=200)

        if not entity and not term:
            return ToolResult(summary="Entity concept lookup requires entity or term.")

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

        async with adapter_locked(adapter):
            rows = await adapter.acypher(cypher, params, access_scope=request.access_scope)

        if entity:
            row0 = (rows or [{}])[0] if isinstance(rows, list) else {}
            candidate_count = int((row0 or {}).get("candidate_count") or 0)
            if candidate_count != 1:
                return ToolResult(
                    summary=(
                        "Entity concept lookup aborted due to ambiguous or missing entity name. "
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
            summary = f"entity_concept: entity={entity} canonical={concept.get('canonical_name') or 'unknown'} aliases={len(concept['aliases'])}"
            label = "entity"
            provenance: Dict[str, Any] = {"entity": entity, "entity_type": entity_type or None, "result": concept}
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
                        "aliases": list(row.get("aliases") or []),
                    }
                )
            summary = f"entity_concepts: term={term} hits={len(results)}"
            label = "term"
            provenance = {"term": term, "results": results}

        chunk_id = derived_chunk_id(tool_name=self.descriptor.name, plan_step=request.plan_step, label=label, content=summary)
        evidence = EvidenceChunk(
            chunk_id=chunk_id,
            source=self.descriptor.name,
            content=summary,
            kind=EVIDENCE_KIND_DERIVED,
            provenance=provenance,
        )
        return ToolResult(summary=summary, evidences=[evidence], diagnostics=provenance)

    @staticmethod
    def _require_adapter(adapter):
        if adapter is None:
            raise RuntimeError("GraphEntityConceptsTool requires a GraphDeepSearchAdapter instance")
        return adapter
