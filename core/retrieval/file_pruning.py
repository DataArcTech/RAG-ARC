"""Candidate pruning utilities for retrieval pipelines.

This module provides a lightweight, deterministic "file-first" pruning strategy:
- Group retrieved chunks by source file.
- Score files using: sum(chunk_scores) / sqrt(n + 1)
- Keep top files, then keep top chunks per file.

Motivation:
- Improves retrieval precision by reducing cross-file noise.
- Reduces reranker/generator context cost by shrinking candidates.

Notes:
- This is NOT a ranking replacement. It is a pre-rerank candidate selector.
- It relies on chunk metadata only (no DB/LLM calls).
"""
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from encapsulation.data_model.schema import Chunk


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def default_file_key(chunk: Chunk) -> str:
    meta = getattr(chunk, "metadata", None) or {}
    for k in ("source_file_id", "sourceFileId", "file_id", "fileId", "document_id", "documentId"):
        token = str(meta.get(k) or "").strip()
        if token:
            return f"file_id:{token}"
    name = str(meta.get("filename") or "").strip()
    if name:
        return f"filename:{name}"
    cid = str(getattr(chunk, "id", "") or "").strip()
    return f"chunk:{cid}" if cid else f"chunk:{id(chunk)}"


def default_chunk_score(chunk: Chunk) -> float:
    """Best-effort numeric relevance score from retrieval metadata.

    We prefer `metadata['score']` because it's used throughout the pipeline,
    but fall back to common HippoRAG sub-scores when available.
    """
    meta = getattr(chunk, "metadata", None) or {}
    for key in ("score", "_hipporag_dense_score", "_hipporag_ppr_score"):
        val = _num(meta.get(key))
        if val is not None:
            return float(val)
    return 0.0


@dataclass(frozen=True, slots=True)
class FilePruneInfo:
    enabled: bool
    chunks_in: int
    chunks_out: int
    files_in: int
    files_kept: int
    max_files: int
    max_chunks_per_file: int
    top_files: List[Dict[str, Any]]


def prune_chunks_by_file(
    chunks: Iterable[Chunk],
    *,
    enabled: bool,
    max_files: int,
    max_chunks_per_file: int,
    file_key_fn: Callable[[Chunk], str] = default_file_key,
    chunk_score_fn: Callable[[Chunk], float] = default_chunk_score,
) -> tuple[List[Chunk], FilePruneInfo]:
    items = [c for c in chunks if isinstance(c, Chunk)]
    if not enabled:
        return (
            items,
            FilePruneInfo(
                enabled=False,
                chunks_in=len(items),
                chunks_out=len(items),
                files_in=0,
                files_kept=0,
                max_files=int(max_files),
                max_chunks_per_file=int(max_chunks_per_file),
                top_files=[],
            ),
        )

    mf = max(1, int(max_files or 1))
    mc = max(1, int(max_chunks_per_file or 1))

    by_file: Dict[str, List[Tuple[float, int, Chunk]]] = {}
    seq = 0
    for ch in items:
        seq += 1
        fk = str(file_key_fn(ch) or "").strip() or f"chunk:{id(ch)}"
        score = float(chunk_score_fn(ch) or 0.0)
        by_file.setdefault(fk, []).append((score, seq, ch))

    file_summaries: List[Tuple[float, str, int]] = []
    for fk, triples in by_file.items():
        # Sort by score desc; keep seq for deterministic tie-break.
        triples_sorted = sorted(triples, key=lambda t: (-t[0], t[1]))
        s = sum(max(0.0, float(t[0])) for t in triples_sorted)
        n = len(triples_sorted)
        node_score = s / math.sqrt(n + 1.0)
        file_summaries.append((float(node_score), fk, n))

    file_summaries.sort(key=lambda t: (-t[0], t[1]))
    kept_files = file_summaries[: min(mf, len(file_summaries))]
    kept_keys = {fk for _score, fk, _n in kept_files}

    out: List[Chunk] = []
    top_files: List[Dict[str, Any]] = []
    for score, fk, n in kept_files:
        triples = by_file.get(fk) or []
        triples_sorted = sorted(triples, key=lambda t: (-t[0], t[1]))
        selected = [t[2] for t in triples_sorted[: min(mc, len(triples_sorted))]]
        out.extend(selected)
        top_files.append({"file_key": fk, "file_score": score, "chunks_in_file": n, "chunks_kept": len(selected)})

    return (
        out,
        FilePruneInfo(
            enabled=True,
            chunks_in=len(items),
            chunks_out=len(out),
            files_in=len(by_file),
            files_kept=len(kept_files),
            max_files=mf,
            max_chunks_per_file=mc,
            top_files=top_files,
        ),
    )

