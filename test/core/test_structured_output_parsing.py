import pytest

from core.knowledge_graph.extraction_models import ExtractedEdges, ExtractedEntities
from core.utils.structured_output import StructuredOutputError, call_pydantic_json_with_retry, parse_pydantic_json_from_llm_text


def test_parse_pydantic_json_from_llm_text_handles_fenced_json() -> None:
    raw = """Here you go:
```json
{"extracted_entities":[{"id":1,"name":"Alice","entity_type":"PERSON"}]}
```"""
    model = parse_pydantic_json_from_llm_text(raw, ExtractedEntities)
    assert model.extracted_entities[0].id == 1
    assert model.extracted_entities[0].name == "Alice"


def test_parse_pydantic_json_from_llm_text_rejects_invalid_payload() -> None:
    raw = """{"extracted_entities":[{"id":0,"name":"","entity_type":""}]}"""
    with pytest.raises(StructuredOutputError):
        parse_pydantic_json_from_llm_text(raw, ExtractedEntities)


def test_edge_relation_type_is_strict_screaming_snake() -> None:
    raw = """{"edges":[{"relation_type":"work for","source_entity_id":1,"target_entity_id":2}]}"""
    with pytest.raises(StructuredOutputError):
        parse_pydantic_json_from_llm_text(raw, ExtractedEdges)


class _AsyncFakeLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        return self.outputs.pop(0)


@pytest.mark.asyncio
async def test_call_pydantic_json_with_retry_repairs_and_validates() -> None:
    llm = _AsyncFakeLLM([
        "not json",
        '{"extracted_entities":[{"id":1,"name":"Alice","entity_type":"PERSON"}]}',
    ])
    model = await call_pydantic_json_with_retry(
        llm_connector=llm,
        messages=[{"role": "user", "content": "extract"}],
        model_cls=ExtractedEntities,
        attempts=2,
    )
    assert model.extracted_entities[0].name == "Alice"
