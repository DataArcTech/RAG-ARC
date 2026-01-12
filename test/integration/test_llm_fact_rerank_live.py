import os

import pytest

from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from core.retrieval.graph_retrieveal.pruned_hipporag_facts import _PrunedHippoRAGFactsMixin


def _env(name: str) -> str:
    return str(os.getenv(name, "") or "").strip()


def _resolve_models() -> list[str]:
    raw = _env("RAGARC_LLM_RERANK_TEST_MODELS")
    if raw:
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        return [p for p in parts if p]
    models = []
    for name in ("CHAT_MODEL_NAME", "OPENAI_CHAT_MODEL", "LOW_COST_MODEL"):
        token = _env(name)
        if token:
            models.append(token)
    # Preserve order and uniqueness.
    out = []
    seen = set()
    for m in models:
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_RAGARC_INTEGRATION_TESTS") != "1",
    reason="integration test opt-in: set RUN_RAGARC_INTEGRATION_TESTS=1",
)
def test_llm_fact_rerank_returns_parseable_json_object() -> None:
    # Require credentials at runtime; do not read .env inside tests.
    if not (_env("CHAT_API_KEY") or _env("OPENAI_API_KEY")):
        pytest.skip("Missing CHAT_API_KEY/OPENAI_API_KEY; set env vars to run live rerank test.")
    if not (_env("CHAT_API_BASE_URL") or _env("OPENAI_BASE_URL")):
        pytest.skip("Missing CHAT_API_BASE_URL/OPENAI_BASE_URL; set env vars to run live rerank test.")

    models = _resolve_models()
    if not models:
        pytest.skip("No models configured; set RAGARC_LLM_RERANK_TEST_MODELS or OPENAI_CHAT_MODEL/LOW_COST_MODEL.")

    llm = OpenAIChatConfig().build()

    class _Runner(_PrunedHippoRAGFactsMixin):
        def __init__(self, llm_client):
            self.llm_client = llm_client

    class _ModelOverrideLLM:
        def __init__(self, inner, model: str):
            self._inner = inner
            self._model = model

        def chat(self, messages, **kwargs):  # noqa: ANN001
            return self._inner.chat(messages, model=self._model, **kwargs)

    candidate_facts = [
        ("Apple", "acquired", "Beats", "owner-x"),
        ("Apple", "released", "iPhone", "owner-x"),
        ("Beats", "is", "a headphone brand", "owner-x"),
        ("Microsoft", "founded_by", "Bill Gates", "owner-x"),
        ("iPhone", "is", "a smartphone", "owner-x"),
    ]
    candidate_indices = list(range(len(candidate_facts)))

    for model in models:
        runner = _Runner(_ModelOverrideLLM(llm, model))
        selected_facts, selected_indices = runner._llm_rerank_filter(
            "Which facts are most relevant to Apple's acquisition of Beats?",
            candidate_facts,
            candidate_indices,
            len_after_rerank=2,
        )
        assert len(selected_facts) == 2
        assert len(selected_indices) == 2
        assert all(isinstance(x, int) for x in selected_indices)

