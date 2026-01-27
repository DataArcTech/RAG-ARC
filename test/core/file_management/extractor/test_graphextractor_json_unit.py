import json
from types import SimpleNamespace

import pytest

from core.file_management.extractor.graphextractor import GraphExtractor
from encapsulation.data_model.schema import Chunk


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        if not self._responses:
            raise RuntimeError("FakeLLM: no more responses configured")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_graphextractor_normalizes_relation_aliases_via_schema(tmp_path) -> None:
    schema_path = tmp_path / "kg_schema.yml"
    schema_path.write_text(
        "\n".join(
            [
                "version: v1",
                "default_domain: default",
                "domains:",
                "  default:",
                "    allowed_relations: [WORKS_AT, RELATES_TO]",
                "    relation_aliases:",
                "      \"work for\": WORKS_AT",
                "    unknown_predicate_policy: collapse",
                "    unknown_predicate_fallback: RELATES_TO",
                "    direction_policy: blacklist",
                "    direction_insensitive_relations: [RELATES_TO]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    entities_json = json.dumps(
        {"extracted_entities": [{"id": 1, "name": "Alice", "entity_type": "PERSON"}, {"id": 2, "name": "Acme", "entity_type": "ORG"}]}
    )
    # LLM emits WORK_FOR; schema alias should normalize this to WORKS_AT.
    edges_json = json.dumps({"edges": [{"relation_type": "WORK_FOR", "source_entity_id": 1, "target_entity_id": 2, "fact": "Alice works for Acme."}]})

    cfg = SimpleNamespace(
        llm_config=None,
        max_concurrent=1,
        error_policy="raise",
        entity_types=None,
        kg_schema_path=str(schema_path),
        schema_prompt_domain=None,
        schema_prompt_max_allowed_relations=50,
        schema_prompt_max_relation_aliases=50,
        language_detection_chinese_ratio_threshold=0.1,
        language_detection_default_language="en",
        edge_reference_time_override="2025-01-01T00:00:00Z",
    )
    extractor = GraphExtractor(cfg)
    extractor.llm = _FakeLLM([entities_json, edges_json])

    chunk = Chunk(id="c1", content="Alice works for Acme.")
    graph = await extractor.extract(chunk)

    assert len(graph.relations) == 1
    assert graph.relations[0][1] == "WORKS_AT"

