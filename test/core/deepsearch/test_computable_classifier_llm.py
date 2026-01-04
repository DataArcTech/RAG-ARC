import asyncio

import pytest

from core.deepsearch.utils.computable_classifier_llm import aclassify_computable_question


class _DummyLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
        return '{"is_computable": true, "reasons": ["has_date"], "suggested_tools": ["graph.latest_truth"]}'


@pytest.mark.asyncio
async def test_classify_computable_question_uses_model_override() -> None:
    llm = _DummyLLM()
    result = await aclassify_computable_question(llm, question="生效日期是什么？", model="cheap-model", temperature=0.0)
    assert result.is_computable is True
    assert result.suggested_tools == ["graph.latest_truth"]
    assert llm.calls
    assert llm.calls[0]["kwargs"].get("model") == "cheap-model"


@pytest.mark.asyncio
async def test_classify_computable_question_rejects_missing_llm() -> None:
    with pytest.raises(RuntimeError):
        await aclassify_computable_question(None, question="x?")

