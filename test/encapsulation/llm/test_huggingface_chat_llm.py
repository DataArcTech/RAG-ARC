from types import SimpleNamespace

import pytest


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        import torch

        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        return "dummy-response"


class _DummyModel:
    def __init__(self):
        import torch

        self.device = torch.device("cpu")

    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, **kwargs):
        import torch

        suffix = torch.tensor([[4, 5, 6]])
        return torch.cat([input_ids, suffix], dim=1)


def test_huggingface_chat_path_uses_generate(monkeypatch):
    from encapsulation.llm.chat.openai import OpenAIChatLLM

    dummy_model = _DummyModel()
    dummy_tokenizer = _DummyTokenizer()

    def _fake_create_transformers_client(_config):
        return dummy_model, dummy_tokenizer

    monkeypatch.setattr(
        "encapsulation.llm.chat.openai.create_transformers_client",
        _fake_create_transformers_client,
    )

    config = SimpleNamespace(
        model_name="dummy",
        max_tokens=8,
        temperature=0.0,
        loading_method="huggingface",
        type="openai_chat",
    )
    llm = OpenAIChatLLM(config)

    output = llm.chat([{"role": "user", "content": "Hello"}])
    assert output == "dummy-response"


def test_huggingface_stream_chat_yields_strings(monkeypatch):
    from encapsulation.llm.chat.openai import OpenAIChatLLM

    dummy_model = _DummyModel()
    dummy_tokenizer = _DummyTokenizer()

    def _fake_create_transformers_client(_config):
        return dummy_model, dummy_tokenizer

    monkeypatch.setattr(
        "encapsulation.llm.chat.openai.create_transformers_client",
        _fake_create_transformers_client,
    )

    config = SimpleNamespace(
        model_name="dummy",
        max_tokens=8,
        temperature=0.0,
        loading_method="huggingface",
        type="openai_chat",
    )
    llm = OpenAIChatLLM(config)

    chunks = list(llm.stream_chat([{"role": "user", "content": "Hello"}], max_tokens=4))
    assert chunks
    assert all(isinstance(chunk, str) for chunk in chunks)


@pytest.mark.asyncio
async def test_huggingface_achat_uses_to_thread(monkeypatch):
    from encapsulation.llm.chat.openai import OpenAIChatLLM

    dummy_model = _DummyModel()
    dummy_tokenizer = _DummyTokenizer()

    def _fake_create_transformers_client(_config):
        return dummy_model, dummy_tokenizer

    monkeypatch.setattr(
        "encapsulation.llm.chat.openai.create_transformers_client",
        _fake_create_transformers_client,
    )

    config = SimpleNamespace(
        model_name="dummy",
        max_tokens=8,
        temperature=0.0,
        loading_method="huggingface",
        type="openai_chat",
    )
    llm = OpenAIChatLLM(config)

    output = await llm.achat([{"role": "user", "content": "Hello"}], max_tokens=4)
    assert output == "dummy-response"


@pytest.mark.integration
@pytest.mark.skipif(
    __import__("os").getenv("RUN_RAGARC_GPT2_CHAT_TESTS") != "1",
    reason="Optional: download tiny-gpt2 into models/gpt2 and set RUN_RAGARC_GPT2_CHAT_TESTS=1 to run.",
)
def test_huggingface_chat_local_gpt2_smoke():
    import os
    from pathlib import Path

    from config.encapsulation.llm.chat.openai import OpenAIChatConfig

    model_path = Path(os.getenv("RAGARC_GPT2_MODEL_DIR", "models/gpt2"))
    if not model_path.exists():
        pytest.skip(f"Local tiny-gpt2 model not found: {model_path} (expected a local directory).")

    old_provider = os.environ.get("CHAT_MODEL_PROVIDER")
    old_model_name = os.environ.get("CHAT_MODEL_NAME")
    os.environ["CHAT_MODEL_PROVIDER"] = "huggingface"
    os.environ["CHAT_MODEL_NAME"] = str(model_path)

    try:
        llm = OpenAIChatConfig(max_tokens=8, temperature=0.0, device="cpu").build()
        out = llm.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(out, str)
        assert out.strip() != ""
    finally:
        if old_provider is None:
            os.environ.pop("CHAT_MODEL_PROVIDER", None)
        else:
            os.environ["CHAT_MODEL_PROVIDER"] = old_provider
        if old_model_name is None:
            os.environ.pop("CHAT_MODEL_NAME", None)
        else:
            os.environ["CHAT_MODEL_NAME"] = old_model_name
