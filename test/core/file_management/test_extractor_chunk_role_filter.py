from types import SimpleNamespace

import pytest

from core.file_management.extractor.base import ExtractorBase
from core.file_management.extractor.metadata_keys import EXTRACTION_SKIPPED_KEY
from encapsulation.data_model.schema import Chunk, GraphData


class _Extractor(ExtractorBase):
    def __init__(self, config):
        super().__init__(config)
        self.calls: list[str] = []

    async def extract(self, chunk: Chunk) -> GraphData:
        self.calls.append(str(chunk.id))
        g = GraphData()
        g.metadata["extracted"] = True
        return g


@pytest.mark.asyncio
async def test_extractor_chunk_role_filter_skips_non_allowed_roles() -> None:
    cfg = SimpleNamespace(
        llm_config=None,
        max_concurrent=4,
        error_policy="attach",
        extraction_cache_enabled=False,
        dedup_by_content=False,
        extract_chunk_roles=["anchor"],
    )
    extractor = _Extractor(cfg)

    anchor = Chunk(content="A", id="c1", metadata={"chunk_role": "anchor"})
    slice_chunk = Chunk(content="B", id="c2", metadata={"chunk_role": "slice"})

    out = await extractor([anchor, slice_chunk])

    assert extractor.calls == ["c1"]
    assert out[0].graph.metadata.get("extracted") is True
    assert out[1].graph.is_empty() is True
    assert out[1].metadata.get(EXTRACTION_SKIPPED_KEY, {}).get("reason") == "chunk_role_filtered"
