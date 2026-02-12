import os

from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from config.encapsulation.llm.parse.vlm_ocr import VLMOcrConfig


def test_openai_chat_base_url_falls_back_when_component_var_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("CHAT_API_BASE_URL", "")
    cfg = OpenAIChatConfig()
    assert cfg.openai_base_url == "https://example.com/v1"


def test_openai_embedding_key_and_base_url_fall_back_when_component_vars_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("EMBEDDING_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "")
    cfg = OpenAIEmbeddingConfig()
    assert cfg.openai_api_key == "sk-test"
    assert cfg.openai_base_url == "https://example.com/v1"


def test_openai_ocr_key_and_base_url_fall_back_when_component_vars_empty(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("OCR_API_KEY", "")
    monkeypatch.setenv("OCR_API_BASE_URL", "")
    # Force openai mode for this unit test.
    monkeypatch.setenv("OCR_MODEL_PROVIDER", "openai")
    cfg = VLMOcrConfig()
    assert cfg.openai_api_key == "sk-test"
    assert cfg.openai_base_url == "https://example.com/v1"


def test_empty_component_env_does_not_hide_openai_env_when_loaded_from_dotenv(monkeypatch, tmp_path):
    # Regression guard: emulate a dotenv-loaded environment where component vars exist but are blank.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("CHAT_API_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_API_BASE_URL", "")
    monkeypatch.setenv("OCR_API_BASE_URL", "")

    assert OpenAIChatConfig().openai_base_url == os.getenv("OPENAI_BASE_URL")
    assert OpenAIEmbeddingConfig().openai_base_url == os.getenv("OPENAI_BASE_URL")
    assert VLMOcrConfig().openai_base_url == os.getenv("OPENAI_BASE_URL")

