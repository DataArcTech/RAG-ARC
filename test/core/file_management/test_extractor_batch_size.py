import pytest

from core.file_management.extractor.base import ExtractorBase
from encapsulation.data_model.schema import Chunk, GraphData


class _StubLLM:
    async def achat(self, _messages):  # noqa: ANN001
        return ""


class _StubLLMConfig:
    def build(self):
        return _StubLLM()


class _Cfg:
    max_concurrent = 10
    batch_size = 2
    error_policy = "raise"
    llm_config = _StubLLMConfig()


class _Extractor(ExtractorBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.seen: list[str] = []

    async def extract(self, chunk: Chunk) -> GraphData:  # noqa: D401
        self.seen.append(str(chunk.id))
        return GraphData()


@pytest.mark.asyncio
async def test_extractor_respects_batch_size_scheduling():
    extractor = _Extractor(_Cfg())
    chunks = [Chunk(id=f"c{i}", content="x") for i in range(5)]
    out = await extractor.extract_concurrent(chunks)

    assert [c.id for c in out] == [f"c{i}" for i in range(5)]
    assert extractor.seen == [f"c{i}" for i in range(5)]
