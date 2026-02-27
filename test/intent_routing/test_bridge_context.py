"""Tests for topic-switch bridge context and INTENT_EMBEDDING_PROVIDER env override."""
import os
from unittest import mock

from application.intent_routing.service import _extract_last_qa_pair


def test_extract_last_qa_pair_basic() -> None:
    messages = [
        ("user", "What are the features of product A?"),
        ("assistant", "Product A has three features: X, Y, Z."),
        ("user", "What about product B?"),
    ]
    result = _extract_last_qa_pair(messages, exclude_text="What about product B?")
    assert "What are the features of product A?" in result
    assert "Product A has three features: X, Y, Z." in result


def test_extract_last_qa_pair_no_assistant_response() -> None:
    messages = [
        ("user", "Hello"),
        ("user", "What about product B?"),
    ]
    result = _extract_last_qa_pair(messages, exclude_text="What about product B?")
    assert "Hello" in result
    assert "assistant" not in result


def test_extract_last_qa_pair_empty_history() -> None:
    result = _extract_last_qa_pair([], exclude_text="anything")
    assert result == ""


def test_extract_last_qa_pair_only_current_query() -> None:
    messages = [("user", "only message")]
    result = _extract_last_qa_pair(messages, exclude_text="only message")
    assert result == ""


def test_intent_embedding_provider_env_override_api() -> None:
    """INTENT_EMBEDDING_PROVIDER=api should resolve to openai_compat."""
    with mock.patch.dict(os.environ, {"INTENT_EMBEDDING_PROVIDER": "api"}, clear=False):
        from config.core.intent_routing.intent_router import load_intent_router_config

        cfg = load_intent_router_config()
        assert cfg.embedding.provider == "openai_compat"


def test_intent_embedding_provider_env_override_local() -> None:
    """INTENT_EMBEDDING_PROVIDER=local should resolve to qwen_local."""
    with mock.patch.dict(os.environ, {"INTENT_EMBEDDING_PROVIDER": "local"}, clear=False):
        from config.core.intent_routing.intent_router import load_intent_router_config

        cfg = load_intent_router_config()
        assert cfg.embedding.provider == "qwen_local"


def test_intent_embedding_provider_default_is_api() -> None:
    """Default (no env var) should resolve to openai_compat (api)."""
    env = dict(os.environ)
    env.pop("INTENT_EMBEDDING_PROVIDER", None)
    with mock.patch.dict(os.environ, env, clear=True):
        from config.core.intent_routing.intent_router import load_intent_router_config

        cfg = load_intent_router_config()
        assert cfg.embedding.provider == "openai_compat"


def test_intent_embedding_api_credential_fallback() -> None:
    """When provider=api and no INTENT_* keys, should fallback to OPENAI_* credentials."""
    env_patch = {
        "INTENT_EMBEDDING_PROVIDER": "api",
        "OPENAI_API_KEY": "test-key-123",
        "OPENAI_BASE_URL": "https://api.example.com/v1",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-test",
    }
    # Remove INTENT-specific keys to force fallback
    env = dict(os.environ)
    for k in ("INTENT_EMBEDDING_API_KEY", "INTENT_EMBEDDING_API_BASE_URL", "INTENT_OPENAI_EMBEDDING_MODEL"):
        env.pop(k, None)
    env.update(env_patch)
    with mock.patch.dict(os.environ, env, clear=True):
        from config.core.intent_routing.intent_router import load_intent_router_config

        cfg = load_intent_router_config()
        assert cfg.embedding.provider == "openai_compat"
        assert cfg.embedding.openai_api_key == "test-key-123"
        assert cfg.embedding.openai_base_url == "https://api.example.com/v1"
        assert cfg.embedding.openai_model_name == "text-embedding-test"
