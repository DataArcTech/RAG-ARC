import pytest

from encapsulation.data_model.deepsearch import GraphQueryContext
from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.tools.base import ToolDescriptor, ToolResult, ToolRunRequest
from core.deepsearch.tools.explore.explore import ExploreTool


class _CaptureTool:
    def __init__(self, name: str):
        self.descriptor = ToolDescriptor(
            name=name,
            channel="graph",
            description="capture",
            namespace=f"test::{name}",
        )
        self.seen_file_scope = None

    async def run(self, request: ToolRunRequest) -> ToolResult:
        meta = request.graph_context.metadata if request.graph_context else {}
        self.seen_file_scope = meta.get("file_scope") if isinstance(meta, dict) else None
        return ToolResult(summary=f"{self.descriptor.name} ok")


@pytest.mark.asyncio
async def test_explore_strips_inherited_file_scope_for_global_actions() -> None:
    web = _CaptureTool("web.search")
    read_pages = _CaptureTool("read.pages")

    explore = ExploreTool(
        llm_connector=None,
        dense_retriever=None,
        bm25_retriever=None,
        tool_overrides={"web.search": web, "read.pages": read_pages},
    )

    ctx = GraphQueryContext(
        adapter_name="stub",
        question="Q",
        access_scope=GraphAccessScope(scope_id="owner"),
        metadata={"file_scope": {"file_ids": ["00000000-0000-0000-0000-000000000001"], "source": "test"}},
    )

    req = ToolRunRequest(
        question="Q",
        plan_step="p1",
        context_evidences=[],
        adapter=None,
        access_scope=ctx.access_scope,
        graph_context=ctx,
        coverage_metrics={},
        extra={
            "actions": [
                {"id": "w1", "tool": "web.search", "args": {"query": "x"}},
                {
                    "id": "r1",
                    "tool": "read.pages",
                    "args": {
                        "file_id": "00000000-0000-0000-0000-000000000001",
                        "page_start": 0,
                        "page_end": 0,
                    },
                },
            ]
        },
    )

    await explore.run(req)
    assert web.seen_file_scope is None
    assert isinstance(read_pages.seen_file_scope, dict)

