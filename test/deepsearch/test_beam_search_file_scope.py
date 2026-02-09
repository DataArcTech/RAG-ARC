import json

import pytest

from core.deepsearch.tools.explore.beam_search import BeamSearchTool
from core.deepsearch.tools import ToolRunRequest
from core.graph_adapter.base import GraphAccessScope, GraphAdapterMetadata, GraphAdapterCapability
from encapsulation.data_model.deepsearch import GraphQueryContext


class _StubLLM:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        return self.outputs.pop(0)


class _CaptureAdapter:
    supports_concurrent_calls = True

    def __init__(self):
        self.last_strategy = None

    async def chain_traverse(self, strategy, *, access_scope=None):  # noqa: ANN001
        self.last_strategy = dict(strategy or {})
        return {
            "strategy": "beam_search",
            "visited": [],
            "hops": int(strategy.get("max_depth") or 1),
            "paths": [
                {
                    "path_id": "path-1",
                    "nodes": ["A", "B"],
                    "triples": [{"head": "A", "relation": "related_to", "tail": "B"}],
                    "score": 1.0,
                    "summary": "A related_to B",
                }
            ],
        }

    def metadata(self) -> GraphAdapterMetadata:
        return GraphAdapterMetadata(
            adapter_name="stub",
            graph_type="stub",
            version="v1",
            capabilities=(GraphAdapterCapability(name="concurrency", modes=(), metrics={"concurrency_safe": True}),),
        )


@pytest.mark.asyncio
async def test_beam_search_includes_file_scope_query_options() -> None:
    llm = _StubLLM([json.dumps([{"path_id": "path-1", "score": 2.0}])])
    tool = BeamSearchTool(llm_connector=llm)
    adapter = _CaptureAdapter()
    req = ToolRunRequest(
        question="q",
        plan_step="p1",
        context_evidences=[],
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        extra={"seed_entities": ["A", "B"]},
        graph_context=GraphQueryContext(
            adapter_name="stub",
            question="q",
            metadata={"file_scope": {"file_ids": ["f1"], "filename_contains": [], "source": "test"}},
            seed_entities=[],
            access_scope=GraphAccessScope(scope_id="owner-1"),
        ),
        coverage_metrics=None,
    )
    result = await tool.run(req)
    assert result.diagnostics.get("file_scope", None) is not None
    opts = (adapter.last_strategy or {}).get("query_options") or {}
    assert isinstance(opts, dict)
    assert opts.get("export_subgraph") is True
    fs = opts.get("file_scope") or {}
    assert fs.get("file_ids") == ["f1"]

