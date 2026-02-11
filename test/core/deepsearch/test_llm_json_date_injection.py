import pytest

from core.deepsearch.utils import llm_json
from core.prompts import runtime_context


class _FakeLLM:
    def __init__(self):
        self.messages = None

    async def achat(self, messages, **_kwargs):  # noqa: ANN001
        self.messages = messages
        return "{}"


@pytest.mark.asyncio
async def test_llm_json_helper_injects_today_into_system_prompt(monkeypatch):
    monkeypatch.setattr(runtime_context, "current_local_date_str", lambda: "2026-02-11")
    llm = _FakeLLM()
    messages = [
        {"role": "system", "content": "Return strict JSON."},
        {"role": "user", "content": "test"},
    ]
    out = await llm_json._call_llm_async(llm, messages)
    assert out == "{}"
    assert llm.messages[0]["content"].startswith("今天是 2026-02-11。\n")
