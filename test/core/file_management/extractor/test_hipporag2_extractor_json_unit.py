import json
from types import SimpleNamespace

import pytest

from core.file_management.extractor.hipporag2_extractor import HippoRAG2Extractor
from encapsulation.data_model.schema import Chunk


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("FakeLLM: no more responses configured")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_hipporag2_extractor_json_two_stage_builds_graphdata() -> None:
    entities_json = json.dumps(
        {
            "extracted_entities": [
                {"id": 1, "name": "Alice", "entity_type": "PERSON"},
                {"id": 2, "name": "Acme Corp", "entity_type": "ORGANIZATION"},
            ]
        }
    )
    edges_json = json.dumps(
        {
            "edges": [
                {
                    "relation_type": "WORKS_AT",
                    "source_entity_id": 1,
                    "target_entity_id": 2,
                    "fact": "Alice works at Acme Corp.",
                    "valid_at": "2025-01-01T00:00:00Z",
                    "invalid_at": None,
                }
            ]
        }
    )

    cfg = SimpleNamespace(
        llm_config=None,
        max_concurrent=1,
        edge_reference_time_override="2025-01-01T00:00:00Z",
        enable_temporal_extraction=False,
        enable_mindmap_extraction=False,
        enable_sdf_extraction=False,
        schema_aware_triple_extraction=False,
        kg_schema_path=None,
        schema_prompt_domain=None,
        schema_prompt_max_allowed_relations=0,
        schema_prompt_max_relation_aliases=0,
        entity_types=None,
    )
    extractor = HippoRAG2Extractor(cfg)
    extractor.llm = _FakeLLM([entities_json, edges_json])

    chunk = Chunk(id="c1", content="Alice works at Acme Corp.")
    graph = await extractor.extract(chunk)

    assert len(graph.entities) == 2
    assert graph.entities[0]["entity_name"] == "Alice"
    assert len(graph.relations) == 1
    assert graph.relations[0][0] == "Alice"
    assert graph.relations[0][1] == "WORKS_AT"
    assert graph.relations[0][2] == "Acme Corp"
    # optional temporal + fact fields
    assert graph.relations[0][3] == "2025-01-01T00:00:00Z"
    assert graph.relations[0][4] is None
    assert "Alice works at Acme" in (graph.relations[0][5] or "")

