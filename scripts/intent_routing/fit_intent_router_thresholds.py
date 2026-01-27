"""Fit semantic-router route thresholds using the official `fit` API.

Reference: https://github.com/aurelio-labs/semantic-router/blob/7b37054d2c41776cc2eb2fdc77b0d322b4945b5c/docs/06-threshold-optimization.ipynb

This script:
- loads `config/core/intent_routing/intent_router.toml` (with env placeholder substitution)
- builds the intent embedding provider (local Qwen or OpenAI-compatible)
- builds a semantic-router `SemanticRouter` via our `SemanticIntentRouter` wrapper
- evaluates and fits per-route thresholds against labeled examples

It prints suggested thresholds; apply them back to TOML via a normal code change
(we intentionally do not rewrite TOML here to avoid fragile "regex patching").
"""

import argparse
import os
import tomllib

from application.intent_routing.embedding import IntentEmbeddingProviderConfig, build_intent_embedding_provider
from config.core.intent_routing.intent_router import load_intent_router_config
from core.intent_routing.semantic_router import IntentRoute, SemanticIntentRouter


def _load_eval_cases(path: str) -> tuple[list[str], list[str | None]]:
    with open(path, "rb") as f:
        payload = tomllib.loads(f.read().decode("utf-8", errors="strict"))
    root = payload.get("intent_router_eval") or {}
    cases = root.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError("intent_router_eval.cases must be a non-empty list")
    X: list[str] = []
    y: list[str | None] = []
    for c in cases:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text") or "").strip()
        intent = c.get("intent")
        label = None if intent is None else (str(intent).strip() or None)
        if not text:
            continue
        X.append(text)
        y.append(label)
    if not X:
        raise ValueError("No valid eval cases loaded")
    return X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=os.getenv("INTENT_ROUTER_CONFIG_PATH", "config/core/intent_routing/intent_router.toml"),
    )
    ap.add_argument("--eval", dest="eval_path", default="config/core/intent_routing/intent_router_eval.toml")
    ap.add_argument("--max-iter", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    # Ensure offline by default for local tests (caller can override).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cfg = load_intent_router_config(args.config)
    emb_cfg = IntentEmbeddingProviderConfig(
        provider=cfg.embedding.provider,
        qwen_model_name=cfg.embedding.qwen_model_name or "",
        qwen_device=cfg.embedding.qwen_device,
        qwen_cache_folder=cfg.embedding.qwen_cache_folder,
        qwen_use_china_mirror=cfg.embedding.qwen_use_china_mirror,
        qwen_model_kwargs=dict(cfg.embedding.qwen_model_kwargs or {}),
        qwen_sentence_transformer_model_kwargs=dict(cfg.embedding.qwen_sentence_transformer_model_kwargs or {}),
        qwen_encode_kwargs=dict(cfg.embedding.qwen_encode_kwargs or {}),
        openai_model_name=cfg.embedding.openai_model_name or "",
        openai_api_key=cfg.embedding.openai_api_key or "",
        openai_base_url=cfg.embedding.openai_base_url or "",
        openai_timeout_seconds=float(cfg.embedding.openai_timeout_seconds),
        openai_max_retries=int(cfg.embedding.openai_max_retries),
        openai_request_batch_size=int(cfg.embedding.openai_request_batch_size),
        openai_supports_batch_input=bool(cfg.embedding.openai_supports_batch_input),
    )
    emb = build_intent_embedding_provider(emb_cfg)

    router = SemanticIntentRouter(
        embedding_provider=emb,
        routes=[IntentRoute(name=r.name, action=r.action, threshold=r.threshold, utterances=r.utterances) for r in cfg.intents],
        default_intent=cfg.default_intent,
        top_k=cfg.top_k,
        aggregation=cfg.aggregation,
    )

    X, y = _load_eval_cases(args.eval_path)

    # Evaluate before fit
    acc_before = router._sr.evaluate(X=X, y=y)  # noqa: SLF001 (intentional: official semantic-router API)
    print(f"[evaluate] before_fit accuracy={acc_before:.4f} n={len(X)}")
    print("[thresholds] before_fit:", router._sr.get_thresholds())  # noqa: SLF001

    router._sr.fit(X=X, y=y, batch_size=int(args.batch_size), max_iter=int(args.max_iter))  # noqa: SLF001
    acc_after = router._sr.evaluate(X=X, y=y)  # noqa: SLF001
    print(f"[evaluate] after_fit accuracy={acc_after:.4f} n={len(X)}")
    print("[thresholds] after_fit:", router._sr.get_thresholds())  # noqa: SLF001

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

