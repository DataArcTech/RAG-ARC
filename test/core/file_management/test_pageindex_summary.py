from types import SimpleNamespace

import pytest

from core.file_management.pageindex.summary import SectionSummaryGenerator
from core.file_management.pageindex.types import SectionNode


class _FlakyLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.config = SimpleNamespace(low_cost_model_name=None)

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        assert messages
        self.calls += 1
        if self.calls == 1:
            raise TypeError("'NoneType' object is not subscriptable")
        return "summary-ok"


@pytest.mark.asyncio
async def test_section_summary_retries_after_transient_failure(monkeypatch):
    monkeypatch.setenv("SECTION_SUMMARY_RETRY_ATTEMPTS", "2")

    llm = _FlakyLLM()
    generator = SectionSummaryGenerator(llm)
    nodes = [
        SectionNode(
            section_id="sec-1",
            file_id="file-1",
            title="Intro",
            path="Intro",
            level=1,
            parent_id=None,
            children=[],
        )
    ]
    chunks_by_section = {"sec-1": [{"content": "This is a long policy text for summary."}]}

    out = await generator.summarize(nodes, chunks_by_section=chunks_by_section)

    assert llm.calls == 2
    assert out["sec-1"] == "summary-ok"
