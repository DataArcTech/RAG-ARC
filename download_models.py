#!/usr/bin/env python3
"""Utility script to pre-download local models into ./models."""

import argparse
import os
from pathlib import Path
from typing import List

# If you need a HuggingFace mirror (e.g., https://hf-mirror.com for China),
# uncomment the next line and set HF_ENDPOINT accordingly.
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download


DEFAULT_EMBEDDING = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_RERANKER = os.getenv("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")
DEFAULT_OCR = os.getenv("DOTS_OCR_MODEL_PATH") or os.getenv("DOTS_OCR_MODEL_NAME") or "rednote-hilab/dots.ocr"
DEFAULT_MINILM = os.getenv("MINILM_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_QWEN_CACHE = os.getenv("RERANKER_CACHE_FOLDER", "./models/Qwen")
DEFAULT_DOTS_CACHE = os.getenv("DOTS_OCR_CACHE_FOLDER", "./models/dots_ocr")
DEFAULT_MINILM_CACHE = os.getenv("MINILM_CACHE_FOLDER", "./models/all-MiniLM-L6-v2")


def _coerce_components(choices: List[str]) -> List[str]:
    if "all" in choices:
        return ["embedding", "reranker", "ocr", "minilm"]
    return choices


def _download_cache(repo_id: str, cache_dir: Path, description: str, use_local_dir: bool = False) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        if use_local_dir:
            local_dir = cache_dir / "model"
            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            target_display = local_dir
        else:
            snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                resume_download=True,
            )
            target_display = cache_dir
        print(f"✅ {description} downloaded → {target_display}")
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to download {description}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local models into ./models for offline usage")
    parser.add_argument(
        "--components",
        nargs="+",
        default=["all"],
        choices=["embedding", "reranker", "ocr", "minilm", "all"],
        help="Choose which model categories to download",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING,
        help="Embedding model repo id (HuggingFace)",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER,
        help="Reranker model repo id (HuggingFace)",
    )
    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR,
        help="DotsOCR model repo id",
    )
    parser.add_argument(
        "--minilm-model",
        default=DEFAULT_MINILM,
        help="MiniLM semantic model repo id",
    )
    parser.add_argument(
        "--qwen-cache",
        default=DEFAULT_QWEN_CACHE,
        help="Cache folder for Qwen-based models (embedding & reranker)",
    )
    parser.add_argument(
        "--dots-cache",
        default=DEFAULT_DOTS_CACHE,
        help="Cache folder for DotsOCR models",
    )
    parser.add_argument(
        "--minilm-cache",
        default=DEFAULT_MINILM_CACHE,
        help="Cache folder for MiniLM models",
    )

    args = parser.parse_args()
    components = _coerce_components(args.components)

    qwen_cache = Path(args.qwen_cache)
    dots_cache = Path(args.dots_cache)
    minilm_cache = Path(args.minilm_cache)

    if "embedding" in components:
        _download_cache(args.embedding_model, qwen_cache, "embedding cache")

    if "reranker" in components:
        _download_cache(args.reranker_model, qwen_cache, "reranker cache")

    if "ocr" in components:
        # DotsOCR uses snapshot_download into cache_folder/model for runtime compatibility
        _download_cache(args.ocr_model, dots_cache, "DotsOCR cache", use_local_dir=True)

    if "minilm" in components:
        _download_cache(args.minilm_model, minilm_cache, "MiniLM cache")


if __name__ == "__main__":
    main()
