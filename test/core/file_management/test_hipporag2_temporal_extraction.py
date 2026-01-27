import pytest

from core.file_management.extractor.hipporag2_extractor import HippoRAG2Extractor
from core.file_management.extractor.metadata_keys import BUSINESS_TIME_KEY
from encapsulation.data_model.schema import Chunk


class _StubLLM:
    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)
        self.calls = 0

    async def achat(self, _messages):  # noqa: ANN001
        if not self._outputs:
            raise AssertionError("No more stub outputs available")
        self.calls += 1
        return self._outputs.pop(0)


class _StubLLMConfig:
    def __init__(self, llm: _StubLLM):
        self._llm = llm

    def build(self):
        return self._llm


class _StubConfig:
    max_concurrent = 1
    entity_types = None
    error_policy = "raise"
    enable_temporal_extraction = True
    enable_mindmap_extraction = False

    def __init__(self, llm: _StubLLM):
        self.llm_config = _StubLLMConfig(llm)


@pytest.mark.asyncio
async def test_temporal_extraction_attaches_business_time_to_chunk_and_graph():
    llm = _StubLLM(
        outputs=[
            '{"extracted_entities":[{"id":1,"name":"远程办公政策","entity_type":"POLICY"},{"id":2,"name":"2024年6月1日","entity_type":"DATE"}]}',
            '{"edges":[{"relation_type":"EFFECTIVE_AT","source_entity_id":1,"target_entity_id":2}]}',
            '{"effective_date":"2024-06-01T00:00:00+00:00","valid_from":"2024-06-01T00:00:00+00:00","valid_to":null,"confidence":0.9}',
        ]
    )
    extractor = HippoRAG2Extractor(_StubConfig(llm))
    chunk = Chunk(id="c1", content="自2024年6月1日起，每周可远程办公2天。")

    graph = await extractor.extract_two_stage(chunk)

    assert BUSINESS_TIME_KEY in chunk.metadata
    assert chunk.metadata[BUSINESS_TIME_KEY]["effective_date"] == "2024-06-01T00:00:00+00:00"
    assert BUSINESS_TIME_KEY in graph.metadata
