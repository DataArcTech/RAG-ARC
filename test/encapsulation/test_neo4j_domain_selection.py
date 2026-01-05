from core.knowledge_graph.schema import schema_from_dict
from encapsulation.data_model.schema import Chunk
from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing import _select_domain_for_chunk


def test_select_domain_falls_back_to_schema_default_domain() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "finance_insurance",
            "domains": {"finance_insurance": {}},
        }
    )
    chunk = Chunk(content="hello", domain=None, metadata={})
    assert _select_domain_for_chunk(chunk, {}, schema) == "finance_insurance"


def test_select_domain_prefers_explicit_chunk_domain() -> None:
    schema = schema_from_dict({"version": "v1", "default_domain": "finance_insurance", "domains": {"default": {}}})
    chunk = Chunk(content="hello", domain="default", metadata={})
    assert _select_domain_for_chunk(chunk, {}, schema) == "default"


def test_select_domain_falls_back_to_metadata_domain() -> None:
    schema = schema_from_dict({"version": "v1", "default_domain": "finance_insurance", "domains": {"finance_insurance": {}}})
    chunk = Chunk(content="hello", domain=None, metadata={"domain": "default"})
    assert _select_domain_for_chunk(chunk, dict(chunk.metadata), schema) == "default"


def test_select_domain_uses_default_when_schema_missing() -> None:
    chunk = Chunk(content="hello", domain=None, metadata={})
    assert _select_domain_for_chunk(chunk, {}, schema=None) == "default"

