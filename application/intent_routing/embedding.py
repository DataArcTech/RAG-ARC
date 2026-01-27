from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from encapsulation.llm.embedding.openai import OpenAIEmbeddingLLM
from encapsulation.llm.embedding.qwen import QwenEmbeddingLLM
from core.intent_routing.semantic_router import EmbeddingProvider


@dataclass(frozen=True)
class IntentEmbeddingProviderConfig:
    provider: Literal["qwen_local", "openai_compat"]
    # qwen_local
    qwen_model_name: str
    qwen_device: str | None
    qwen_cache_folder: str | None
    qwen_use_china_mirror: bool
    # Passed to SentenceTransformer(...) via QwenEmbeddingLLM (advanced tuning).
    qwen_model_kwargs: dict[str, Any]
    # Passed to SentenceTransformer(..., model_kwargs=...) via QwenEmbeddingLLM.
    qwen_sentence_transformer_model_kwargs: dict[str, Any]
    # Passed to SentenceTransformer.encode(...) via QwenEmbeddingLLM (batch size, etc).
    qwen_encode_kwargs: dict[str, Any]
    # openai_compat
    openai_model_name: str
    openai_api_key: str
    openai_base_url: str
    openai_timeout_seconds: float
    openai_max_retries: int
    openai_request_batch_size: int
    openai_supports_batch_input: bool


def build_intent_embedding_provider(cfg: IntentEmbeddingProviderConfig) -> EmbeddingProvider:
    """Build an embedding provider for intent routing (decoupled from RAG embeddings)."""
    if cfg.provider == "qwen_local":
        # QwenEmbeddingLLM expects a config-like object with attribute access.
        config_obj = SimpleNamespace(
            model_name=cfg.qwen_model_name,
            device=cfg.qwen_device or "cpu",
            cache_folder=cfg.qwen_cache_folder,
            use_china_mirror=bool(cfg.qwen_use_china_mirror),
            loading_method="huggingface",
            model_kwargs=dict(cfg.qwen_model_kwargs or {}),
            sentence_transformer_model_kwargs=dict(cfg.qwen_sentence_transformer_model_kwargs or {}),
            encode_kwargs=dict(cfg.qwen_encode_kwargs or {}),
            type="qwen_embedding",
        )
        model = QwenEmbeddingLLM(config_obj)
        return _EmbeddingProviderAdapter(model)

    if cfg.provider == "openai_compat":
        config_obj = SimpleNamespace(
            model_name=cfg.openai_model_name,
            embedding_dimensions=None,
            loading_method="openai",
            openai_api_key=cfg.openai_api_key,
            openai_base_url=cfg.openai_base_url,
            organization=None,
            timeout=float(cfg.openai_timeout_seconds),
            max_retries=int(cfg.openai_max_retries),
            request_batch_size=int(cfg.openai_request_batch_size),
            supports_batch_input=bool(cfg.openai_supports_batch_input),
            rate_limit_max_retries=0,
            rate_limit_default_sleep_seconds=0.0,
            rate_limit_max_sleep_seconds=0.0,
            type="openai_embedding",
        )
        model = OpenAIEmbeddingLLM(config_obj)
        return _EmbeddingProviderAdapter(model)

    raise ValueError(f"Unsupported intent embedding provider: {cfg.provider}")


class _EmbeddingProviderAdapter:
    def __init__(self, model: Any) -> None:
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = self._model.embed(texts)
        # EmbeddingLLM returns list[list[float]] for list inputs.
        if not isinstance(out, list) or (out and not isinstance(out[0], list)):
            raise RuntimeError("Unexpected embedding output shape")
        return out  # type: ignore[return-value]
