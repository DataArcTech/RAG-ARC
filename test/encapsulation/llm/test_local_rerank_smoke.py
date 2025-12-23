import os
from pathlib import Path

import pytest

from encapsulation.data_model.schema import Chunk
from config.encapsulation.llm.rerank.qwen import QwenRerankConfig


def _estimate_weight_bytes(model_dir: Path) -> int:
    total = 0
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        for file in model_dir.rglob(pattern):
            try:
                total += file.stat().st_size
            except OSError:
                continue
    return total


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_RAGARC_LOCAL_RERANK_TESTS") != "1",
    reason="Optional: set RUN_RAGARC_LOCAL_RERANK_TESTS=1 to run (requires local reranker weights).",
)
def test_local_qwen_reranker_smoke():
    snapshot_root = Path(
        os.getenv(
            "RAGARC_RERANKER_SNAPSHOTS",
            "models/Qwen/models--Qwen--Qwen3-Reranker-0.6B/snapshots",
        )
    )
    if not snapshot_root.exists():
        pytest.skip(f"Reranker snapshots not found: {snapshot_root}")

    snapshot_dirs = [p for p in snapshot_root.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        pytest.skip(f"No snapshot directories under: {snapshot_root}")

    model_dir = snapshot_dirs[0]
    weight_bytes = _estimate_weight_bytes(model_dir)
    if weight_bytes > 800_000_000 and os.getenv("RAGARC_ALLOW_LARGE_MODELS") != "1":
        pytest.skip(
            f"Reranker weights are large ({weight_bytes} bytes); set RAGARC_ALLOW_LARGE_MODELS=1 to force loading."
        )

    device = os.getenv("RERANKER_DEVICE", "cpu")
    reranker = QwenRerankConfig(model_name=str(model_dir), device=device, cache_folder=None).build()

    chunks = [
        Chunk(id="c1", content="Machine learning is a field of AI.", metadata={}),
        Chunk(id="c2", content="This is a recipe for pasta.", metadata={}),
        Chunk(id="c3", content="Neural networks learn representations.", metadata={}),
    ]
    results = reranker.rerank("What is machine learning?", chunks, top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    for idx, score in results:
        assert isinstance(idx, int)
        assert 0 <= idx < len(chunks)
        assert isinstance(score, float)
