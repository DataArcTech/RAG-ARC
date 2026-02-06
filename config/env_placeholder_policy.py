"""Central policy for JSON `${ENV_VAR}` placeholder substitution.

Goals:
- Keep defaults centrally managed under `config/` (not scattered across runtime call sites).
- Avoid noisy startup warnings for optional placeholders when a config-level default exists.
- Preserve the existing behavior for required secrets/endpoints: missing values should still be visible
  and ultimately fail fast when building the corresponding components.

Policy:
1) If the env var is set and non-empty: use it.
2) Else if the env var appears in `ENV_DEFAULTS`: substitute that default string.
3) Else if the env var appears in `SILENT_MISSING_ENV_VARS`: omit the key (for full placeholders) and
   do not warn. Downstream config defaults (Pydantic) should handle it.
4) Otherwise: keep existing Register behavior (warn and omit/keep placeholder depending on context).
"""
# Defaults are intentionally limited to non-secret, non-endpoint values.
# Secrets and API endpoints should be configured via env and validated by the corresponding config builders.

ENV_DEFAULTS: dict[str, str] = {
    # Knowledge / storage paths (three distinct stores; cannot be inferred from LocalDBConfig alone).
    "FILE_STORE_BASE_PATH": "./data/file_store",
    "PARSED_CONTENT_STORE_BASE_PATH": "./data/parsed_content_store",
    "CHUNK_STORE_BASE_PATH": "./data/chunk_store",
    # Parser output root (controls native/dots_ocr/vlm_ocr/mineru subfolders).
    "PARSER_OUTPUT_DIR": "./data/parsed_files",
    # MultiPath fusion weights (RAG inference).
    # Order aligns with `config/json_configs/rag_inference*.json` retrievers: [dense, bm25, graph].
    "RAG_RETRIEVAL_WEIGHT_DENSE": "1.0",
    "RAG_RETRIEVAL_WEIGHT_BM25": "1.0",
    "RAG_RETRIEVAL_WEIGHT_GRAPH": "1.5",
    # Intent router (semantic intent classification)
    "INTENT_OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
    "INTENT_QWEN_EMBEDDING_MODEL_NAME": "Qwen/Qwen3-Embedding-0.6B",
    "INTENT_EMBEDDING_DEVICE": "cpu",
    # Local intent embedding cache (aligns with default local model layout under ./models).
    "INTENT_EMBEDDING_CACHE_FOLDER": "./models/Qwen",
}

# Missing placeholders that should not produce warnings because downstream config has safe defaults.
SILENT_MISSING_ENV_VARS: set[str] = {
    # Index paths (config classes already default to ./data/*).
    "GRAPH_STORAGE_PATH",
    "GRAPH_INDEX_NAME",
    "FAISS_INDEX_PATH",
    "BM25_INDEX_PATH",
    # Optional embedding dimension override (auto-detect / model defaults).
    "EMBEDDING_DIMENSIONS",
    # DeepSearch optional env knobs (Python config provides defaults).
    "DEEPSEARCH_WEB_PROVIDER",
    "DEEPSEARCH_PLAN_OUTPUT_DIR",
    "DEEPSEARCH_TOOL_ARTIFACT_DIR",
    "DEEPSEARCH_TOOL_AUDIT_LABEL",
    "DEEPSEARCH_TOOL_MCP_AUDIT_LABEL",
    "DEEPSEARCH_EXTERNAL_CACHE_DIR",
    "DEEPSEARCH_DEFAULT_ADAPTER",
    # Local model cache/device overrides.
    "DOTS_OCR_CACHE_FOLDER",
    "EMBEDDING_CACHE_FOLDER",
    "EMBEDDING_DEVICE",
    "EMBEDDING_MODEL_NAME",
    "DEVICE",
    "RERANKER_CACHE_FOLDER",
    "RERANKER_MODEL_NAME",
    # Semantic unit chunker override knobs (config has defaults).
    "SEMANTIC_CHUNKING_LEVEL",
    "TABLE_SMALL_MAX_TOKENS",
    "TABLE_SLICE_MAX_TOKENS",
    "TABLE_SLICE_OVERLAP_ROWS",
    "CODE_SMALL_MAX_TOKENS",
    "CODE_SLICE_MAX_TOKENS",
    "CODE_SLICE_OVERLAP_LINES",
    "LIST_SMALL_MAX_TOKENS",
    "LIST_SLICE_MAX_TOKENS",
    "LIST_SLICE_OVERLAP_ITEMS",
    # Dependency services: local defaults exist in config, and dependency health checks run at startup.
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_PASSWORD",
    "NEO4J_URL",
    "NEO4J_USERNAME",
    "NEO4J_DATABASE",
    # Model names (config provides reasonable defaults).
    "OPENAI_CHAT_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_OCR_MODEL",
    # Intent router optional overrides (TOML can provide defaults).
    "INTENT_ROUTER_CONFIG_PATH",
    "INTENT_EMBEDDING_API_KEY",
    "INTENT_EMBEDDING_API_BASE_URL",
    "INTENT_OPENAI_EMBEDDING_MODEL",
    "INTENT_QWEN_EMBEDDING_MODEL_NAME",
    "INTENT_EMBEDDING_DEVICE",
    "INTENT_EMBEDDING_CACHE_FOLDER",
    # Optional external search (only required when enabled).
    "TAVILY_API_KEY",
    # Chunking knobs (TokenChunkerConfig has safe defaults).
    "TOKEN_CHUNK_SIZE",
    "TOKEN_CHUNK_OVERLAP",
}
