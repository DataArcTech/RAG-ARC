import pytest

from core.deepsearch.utils.llm_json import call_llm_json_with_retry


class _FakeLLM:
    def __init__(self, responses):  # noqa: ANN001
        self._responses = list(responses)

    async def achat(self, messages, **kwargs):  # noqa: ANN001, ARG002
        # Pop sequentially to simulate retries.
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


@pytest.mark.asyncio
async def test_call_llm_json_with_retry_recovers_from_incomplete_json() -> None:
    llm = _FakeLLM(
        [
            "```json\n{\"terms\": [",  # truncated / invalid
            "{\"terms\": [{\"text\": \"后备受保人\", \"kind\": \"role\"}]}",  # valid
        ]
    )
    payload = await call_llm_json_with_retry(
        llm_connector=llm,
        messages=[{"role": "system", "content": "Return JSON only."}, {"role": "user", "content": "Query: ..."}],
        expected="object",
        temperature=0.0,
        max_tokens=120,
        attempts=2,
    )
    assert isinstance(payload, dict)
    assert isinstance(payload.get("terms"), list)
    assert payload["terms"][0]["text"] == "后备受保人"
