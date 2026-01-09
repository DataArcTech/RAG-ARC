"""PrunedHippoRAG Neo4j indexing mixins (composed to keep modules maintainable)."""

from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing_ingest import (
    _PrunedHippoRAGNeo4jIndexingIngestMixin,
    _coerce_fact_provenance_max_source_chunks,
    _select_domain_for_chunk,
)
from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing_ops import _PrunedHippoRAGNeo4jIndexingOpsMixin


class _PrunedHippoRAGNeo4jIndexingMixin(
    _PrunedHippoRAGNeo4jIndexingIngestMixin,
    _PrunedHippoRAGNeo4jIndexingOpsMixin,
):
    """Combined mixin for PrunedHippoRAGNeo4jStore (indexing + ingestion)."""


__all__ = [
    "_PrunedHippoRAGNeo4jIndexingMixin",
    "_coerce_fact_provenance_max_source_chunks",
    "_select_domain_for_chunk",
]

