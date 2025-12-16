import os
from pathlib import Path

import pytest

from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_RAGARC_LOCAL_EMBEDDING_TESTS") != "1",
    reason="Optional: set RUN_RAGARC_LOCAL_EMBEDDING_TESTS=1 (and download a local sentence-transformers model) to run.",
)
def test_local_sentence_transformer_embedding_smoke():
    snapshot_root = Path(
        os.getenv(
            "RAGARC_ST_MODEL_SNAPSHOTS",
            "models/all-MiniLM-L6-v2/models--sentence-transformers--all-MiniLM-L6-v2/snapshots",
        )
    )
    if not snapshot_root.exists():
        pytest.skip(f"SentenceTransformer snapshots not found: {snapshot_root}")

    snapshot_dirs = [p for p in snapshot_root.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        pytest.skip(f"No snapshot directories under: {snapshot_root}")

    model_dir = snapshot_dirs[0]

    old_provider = os.environ.get("EMBEDDING_MODEL_PROVIDER")
    old_model_name = os.environ.get("OPENAI_EMBEDDING_MODEL")
    os.environ["EMBEDDING_MODEL_PROVIDER"] = "huggingface"
    os.environ["OPENAI_EMBEDDING_MODEL"] = str(model_dir)

    try:
        embedder = OpenAIEmbeddingConfig().build()
        vec = embedder.embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) > 0
        assert all(isinstance(x, float) for x in vec)
    finally:
        if old_provider is None:
            os.environ.pop("EMBEDDING_MODEL_PROVIDER", None)
        else:
            os.environ["EMBEDDING_MODEL_PROVIDER"] = old_provider

        if old_model_name is None:
            os.environ.pop("OPENAI_EMBEDDING_MODEL", None)
        else:
            os.environ["OPENAI_EMBEDDING_MODEL"] = old_model_name
