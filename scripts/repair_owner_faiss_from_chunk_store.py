#!/usr/bin/env python3
"""Repair an owner-scoped FAISS index by (re-)adding missing chunks from chunk_store.

Why this exists:
- In practice, indexing runs can fail transiently (network/provider). FAISS can end up
  incomplete even though the file status is marked INDEXED.
- For DeepSearch, incomplete dense indexes cause severe navigation/retrieval regressions.

This script is intentionally owner-scoped and idempotent:
- It loads the existing FAISS index for the owner (if present),
- scans `data/localdb/chunk_store/chunks/**.json` for the owner's chunks,
- adds only chunk_ids missing from FAISS docstore,
- saves the updated index back to the same owner-scoped directory.

Notes:
- Uses API-based embeddings (OpenAI-compatible) per `RAG-ARC/.env`. No local models.
- Does NOT touch BM25 or graph indexes (avoids duplication/pollution).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_STORE_ROOT = REPO_ROOT / "data" / "localdb" / "chunk_store" / "chunks"


def _iter_chunk_json_paths() -> Iterator[Path]:
    if not CHUNK_STORE_ROOT.exists():
        return iter(())
    yield from CHUNK_STORE_ROOT.rglob("*.json")


def _load_chunk_payload(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--owner-id", required=True, help="Owner UUID (tenant/user).")
    p.add_argument(
        "--file-id",
        action="append",
        default=[],
        help="Optional file UUID filter (repeatable). If unset, repairs all files for the owner.",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Embedding/update batch size.")
    p.add_argument("--dry-run", action="store_true", help="Scan and report only; do not write.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Ensure we read the same credentials/config as the service/CLI.
    load_dotenv(REPO_ROOT / ".env", override=True)
    os.environ.setdefault("MODEL_PROFILE", "api")

    owner_id = str(args.owner_id).strip()
    file_ids = {str(x).strip() for x in (args.file_id or []) if str(x).strip()}
    batch_size = max(1, int(args.batch_size))

    from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
    from encapsulation.data_model.schema import Chunk
    from encapsulation.database.index_scoping import owner_scoped_dir
    from encapsulation.database.vector_db.faiss import FaissVectorDB

    cfg = FaissVectorDBConfig(
        embedding_config={
            "type": "openai_embedding",
            "model_name": os.getenv("OPENAI_EMBEDDING_MODEL"),
            "embedding_dimensions": os.getenv("EMBEDDING_DIMENSIONS") or None,
        }
    )

    owner_index_dir = owner_scoped_dir(
        cfg.index_path,
        owner_id=owner_id,
        owner_dirname=cfg.owner_scoped_dirname,
        global_owner_name=cfg.owner_scoped_global_owner_name,
    )
    db = FaissVectorDB(cfg.model_copy(update={"index_path": owner_index_dir}))
    try:
        db.load_index(owner_index_dir)
    except Exception:
        # Missing/invalid index: start from empty; update_index() will create a new index.
        pass

    existing_ids = set(getattr(db, "docstore", {}) or {})

    missing_chunks: list[Chunk] = []
    scanned = 0
    matched_owner = 0
    matched_files = 0

    for path in _iter_chunk_json_paths():
        scanned += 1
        payload = _load_chunk_payload(path)
        if not isinstance(payload, dict):
            continue

        src = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
        if str(src.get("owner_id") or "").strip() != owner_id:
            continue
        matched_owner += 1

        source_file_id = str(src.get("source_file_id") or "").strip()
        if file_ids and source_file_id not in file_ids:
            continue
        matched_files += 1

        chunk_id = path.stem
        if not chunk_id or chunk_id in existing_ids:
            continue

        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        meta = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), dict) else {}
        meta.update(src)
        missing_chunks.append(
            Chunk(
                id=chunk_id,
                content=content,
                owner_id=owner_id,
                metadata=meta,
            )
        )

    print(
        json.dumps(
            {
                "owner_id": owner_id,
                "owner_index_dir": owner_index_dir,
                "scanned_chunk_json_files": scanned,
                "matched_owner": matched_owner,
                "matched_file_filter": matched_files,
                "existing_faiss_docstore": len(existing_ids),
                "missing_chunks": len(missing_chunks),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=True,
        )
    )

    if args.dry_run or not missing_chunks:
        return 0

    # Update in batches to avoid provider timeouts on huge embedding payloads.
    added = 0
    for i in range(0, len(missing_chunks), batch_size):
        batch = missing_chunks[i : i + batch_size]
        ids = db.update_index(batch) or []
        added += len(ids)

    db.save_index(owner_index_dir)

    print(
        json.dumps(
            {
                "owner_id": owner_id,
                "added_chunk_ids": added,
                "new_faiss_docstore": len(getattr(db, "docstore", {}) or {}),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

