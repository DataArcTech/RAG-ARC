import pytest

from core.deepsearch.tools import ToolRunRequest
from core.deepsearch.tools.explore.read_neighbors import ReadNeighborsTool
from core.graph_adapter.base import GraphAccessScope


class _FakeAdapter:
    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ANN001, ARG002
        params = params or {}
        if "WHERE c.chunk_id = $chunk_id" in cypher:
            # Center lookup.
            if params.get("chunk_id") != "c2":
                return []
            return [
                {
                    "chunk_id": "c2",
                    "source_file_id": "file-1",
                    "metadata": {"chunk_index": 2, "source_file_id": "file-1"},
                }
            ]

        if "WHERE c.source_file_id = $file_id" in cypher:
            if params.get("file_id") != "file-1":
                return []
            return [
                {"chunk_id": "c0", "content": "zero", "metadata": {"chunk_index": 0, "source_file_id": "file-1"}},
                {"chunk_id": "c1", "content": "one", "metadata": {"chunk_index": 1, "source_file_id": "file-1"}},
                {"chunk_id": "c2", "content": "two", "metadata": {"chunk_index": 2, "source_file_id": "file-1"}},
                {"chunk_id": "c3", "content": "three", "metadata": {"chunk_index": 3, "source_file_id": "file-1"}},
                {"chunk_id": "c4", "content": "four", "metadata": {"chunk_index": 4, "source_file_id": "file-1"}},
            ]

        return []


@pytest.mark.asyncio
async def test_read_neighbors_reads_window() -> None:
    tool = ReadNeighborsTool()
    req = ToolRunRequest(
        question="read",
        plan_step="plan_01",
        context_evidences=[],
        adapter=_FakeAdapter(),
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"chunk_id": "c2", "before": 1, "after": 2, "max_chars": 10000, "max_chunks": 20},
        graph_context=None,
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert "read.neighbors returned" in result.summary
    assert len(result.evidences) == 1
    evidence = result.evidences[0]
    assert evidence.source == "read.neighbors"
    assert "one" in evidence.content
    assert "two" in evidence.content
    assert "three" in evidence.content
    assert "four" in evidence.content
    assert "zero" not in evidence.content

