from types import SimpleNamespace

import pytest

from core.file_management.extractor.base import ExtractorBase
from encapsulation.data_model.schema import Chunk, GraphData
from core.file_management.extractor.hipporag2_extractor import HippoRAG2Extractor


class _LLMConfigStub:
    def build(self):
        return object()


class _ExplodingExtractor(ExtractorBase):
    async def extract(self, chunk: Chunk) -> GraphData:  # noqa: ARG002
        raise RuntimeError("boom")


class _LLMRaisesStub:
    async def achat(self, messages):  # noqa: ARG002
        raise RuntimeError("llm_down")


class _LLMConfigRaisesStub:
    def build(self):
        return _LLMRaisesStub()


@pytest.mark.asyncio
async def test_extractor_attach_error_policy_marks_chunk_and_graph_metadata() -> None:
    cfg = SimpleNamespace(llm_config=_LLMConfigStub(), max_concurrent=1, error_policy="attach")
    extractor = _ExplodingExtractor(cfg)
    chunk = Chunk(content="hello", id="chunk-1")

    (out,) = await extractor([chunk])

    assert out.graph.is_empty() is True
    assert out.graph.metadata.get("extraction_error") is not None
    assert out.metadata.get("extraction_error") is not None


@pytest.mark.asyncio
async def test_extractor_raise_error_policy_propagates_exception() -> None:
    cfg = SimpleNamespace(llm_config=_LLMConfigStub(), max_concurrent=1, error_policy="raise")
    extractor = _ExplodingExtractor(cfg)

    with pytest.raises(RuntimeError, match="boom"):
        await extractor([Chunk(content="hello", id="chunk-1")])


@pytest.mark.asyncio
async def test_hipporag2_extractor_llm_failure_is_observable_via_error_metadata() -> None:
    cfg = SimpleNamespace(llm_config=_LLMConfigRaisesStub(), max_concurrent=1, error_policy="attach")
    extractor = HippoRAG2Extractor(cfg)
    (out,) = await extractor([Chunk(content="hello", id="chunk-1")])
    assert out.graph.is_empty() is True
    assert out.graph.metadata.get("extraction_error", {}).get("message") == "llm_down"
