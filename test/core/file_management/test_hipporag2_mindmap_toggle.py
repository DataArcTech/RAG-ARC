import pytest

from core.file_management.extractor.hipporag2_extractor import HippoRAG2Extractor
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
    enable_temporal_extraction = False

    def __init__(self, llm: _StubLLM, *, enable_mindmap_extraction: bool):
        self.llm_config = _StubLLMConfig(llm)
        self.enable_mindmap_extraction = enable_mindmap_extraction


@pytest.mark.asyncio
async def test_mindmap_extraction_is_disabled_by_default() -> None:
    llm = _StubLLM(
        outputs=[
            '{"extracted_entities":[{"id":1,"name":"平安保险","entity_type":"COMPANY"},{"id":2,"name":"条款","entity_type":"CONCEPT"}]}',
            '{"edges":[{"relation_type":"HAS_POLICY","source_entity_id":1,"target_entity_id":2}]}',
        ]
    )
    extractor = HippoRAG2Extractor(_StubConfig(llm, enable_mindmap_extraction=False))
    chunk = Chunk(id="c1", content="hello")
    await extractor.extract_two_stage(chunk)
    assert "mindmap" not in chunk.metadata
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_mindmap_extraction_can_be_enabled() -> None:
    llm = _StubLLM(
        outputs=[
            '{"extracted_entities":[{"id":1,"name":"平安保险","entity_type":"COMPANY"},{"id":2,"name":"条款","entity_type":"CONCEPT"}]}',
            '{"edges":[{"relation_type":"HAS_POLICY","source_entity_id":1,"target_entity_id":2}]}',
            "### MINDMAP\n1\t[concept] 条款\n",
        ]
    )
    extractor = HippoRAG2Extractor(_StubConfig(llm, enable_mindmap_extraction=True))
    chunk = Chunk(id="c1", content="hello")
    await extractor.extract_two_stage(chunk)
    assert isinstance(chunk.metadata.get("mindmap"), dict)
    assert llm.calls == 3
