import pytest

from core.prompts.llm_json import JSON_RETRY_INSTRUCTION_EN
from core.utils.llm_json import (
    call_llm_json_with_retry,
    call_llm_json_with_retry_sync,
    repair_json_from_raw_with_retry,
    repair_json_from_raw_with_retry_sync,
    call_prompt_json_with_retry_sync,
)


class _AsyncFakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def achat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self._outputs.pop(0)


class _SyncFakeLLM:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self._outputs.pop(0)


@pytest.mark.asyncio
async def test_call_llm_json_with_retry_async_retries_and_parses() -> None:
    llm = _AsyncFakeLLM(["not json", '{"ok": true}'])
    payload = await call_llm_json_with_retry(
        llm_connector=llm,
        messages=[{"role": "user", "content": "hi"}],
        expected="dict",
        attempts=2,
        temperature=0.0,
        max_tokens=64,
    )
    assert payload == {"ok": True}
    assert len(llm.calls) == 2
    assert JSON_RETRY_INSTRUCTION_EN in llm.calls[1]["messages"][-1]["content"]


def test_call_llm_json_with_retry_sync_retries_and_parses() -> None:
    llm = _SyncFakeLLM(["not json", '{"ok": true}'])
    payload = call_llm_json_with_retry_sync(
        llm_connector=llm,
        messages=[{"role": "user", "content": "hi"}],
        expected="dict",
        attempts=2,
        temperature=0.0,
        max_tokens=64,
    )
    assert payload == {"ok": True}
    assert len(llm.calls) == 2
    assert JSON_RETRY_INSTRUCTION_EN in llm.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_repair_json_from_raw_with_retry_async() -> None:
    llm = _AsyncFakeLLM(["still bad", '{"fixed": 1}'])
    payload = await repair_json_from_raw_with_retry(
        llm_connector=llm,
        messages=[{"role": "user", "content": "hi"}],
        raw="broken",
        expected="dict",
        attempts=2,
        temperature=0.0,
    )
    assert payload == {"fixed": 1}
    assert len(llm.calls) == 2
    assert llm.calls[0]["messages"][1]["role"] == "assistant"


def test_repair_json_from_raw_with_retry_sync() -> None:
    llm = _SyncFakeLLM(["still bad", '{"fixed": 1}'])
    payload = repair_json_from_raw_with_retry_sync(
        llm_connector=llm,
        messages=[{"role": "user", "content": "hi"}],
        raw="broken",
        expected="dict",
        attempts=2,
        temperature=0.0,
    )
    assert payload == {"fixed": 1}
    assert len(llm.calls) == 2
    assert llm.calls[0]["messages"][1]["role"] == "assistant"



def test_call_prompt_json_with_retry_sync_retries_and_parses() -> None:
    outputs = ["not json", '{"ok": true}']
    seen_prompts = []

    def _infer_once(prompt: str) -> str:
        seen_prompts.append(prompt)
        return outputs.pop(0)

    payload = call_prompt_json_with_retry_sync(
        infer_once=_infer_once,
        prompt="Return JSON",
        expected="dict",
        attempts=2,
        return_raw=False,
    )
    assert payload == {"ok": True}
    assert len(seen_prompts) == 2
    assert JSON_RETRY_INSTRUCTION_EN in seen_prompts[1]
    assert "Previous output" in seen_prompts[1]
