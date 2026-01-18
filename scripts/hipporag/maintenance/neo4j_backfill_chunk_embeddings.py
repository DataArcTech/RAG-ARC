#!/usr/bin/env python3
"""
Backfill missing chunk embeddings for PrunedHippoRAG Neo4j store.

Graph retrieval uses `${GRAPH_STORAGE_PATH}/${GRAPH_INDEX_NAME}_chunk_embeddings.pkl` as its dense
fallback + passage-weighting signal. If Chunk nodes are recreated (new chunk_id UUIDs) but the
pickle isn't updated, missing chunks will be treated as zero vectors at retrieval time, which can
destabilize dense signals and downstream graph scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from framework.register import Register


def _load_dotenv(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _as_uuid(value: str) -> uuid.UUID:
    return uuid.UUID(str(value))


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill missing Neo4j graph chunk embeddings.")
    ap.add_argument("--dotenv", default=".env")
    ap.add_argument("--rag-config", default="config/json_configs/rag_inference.json")
    ap.add_argument("--owner-id", required=True, type=_as_uuid)
    ap.add_argument("--source-file-id", default="", help="Optional: only backfill chunks for this file_id.")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap for generated embeddings (0=no cap).")
    ap.add_argument("--dry-run", action="store_true", help="Only report missing ids; do not write.")
    args = ap.parse_args()

    _load_dotenv(args.dotenv)

    from config.application.rag_inference_config import RAGInferenceConfig

    cfg_data = json.loads(Path(args.rag_config).read_text(encoding="utf-8", errors="replace"))
    cfg_data = Register()._substitute_env_vars(cfg_data)  # type: ignore[attr-defined]
    rag_cfg = RAGInferenceConfig.model_validate(cfg_data)

    graph_cfg = None
    for r in rag_cfg.retrieval_config.retrievers:
        if getattr(r, "type", None) == "pruned_hipporag_neo4j_retrieval":
            graph_cfg = r
            break
    if graph_cfg is None:
        raise SystemExit("No pruned_hipporag_neo4j_retrieval found in rag_inference config")

    store = graph_cfg.graph_config.build()
    owner = str(args.owner_id)
    fid = str(args.source_file_id or "").strip()

    if fid:
        query = "MATCH (c:Chunk {owner_id:$owner, source_file_id:$fid}) RETURN c.chunk_id AS chunk_id"
        rows = store._execute_query(query, {"owner": owner, "fid": fid}) or []
    else:
        query = "MATCH (c:Chunk {owner_id:$owner}) RETURN c.chunk_id AS chunk_id"
        rows = store._execute_query(query, {"owner": owner}) or []

    chunk_ids = [str(r.get("chunk_id") or "").strip() for r in rows if isinstance(r, dict)]
    chunk_ids = [cid for cid in chunk_ids if cid]

    emb = getattr(store, "chunk_embeddings", {}) or {}
    missing = [cid for cid in chunk_ids if cid not in emb]
    if args.limit and int(args.limit) > 0:
        missing = missing[: int(args.limit)]

    print("# Neo4j chunk embeddings backfill")
    print(f"owner_id={owner}")
    if fid:
        print(f"source_file_id={fid}")
    print(f"chunks_total={len(chunk_ids)} chunk_embeddings_cached={len(emb)} missing={len(missing)}")

    if not missing or bool(args.dry_run):
        return 0

    store.batch_generate_embeddings(chunk_ids=missing)
    store.save_index(store.storage_path, store.index_name)
    print(f"done: generated={len(missing)} saved_to={store.storage_path}/{store.index_name}_chunk_embeddings.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

