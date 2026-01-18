#!/usr/bin/env python3
"""
Repair Neo4j `RELATES_TO.source_chunk_ids` provenance for a single owner (dry-run by default).

Problem
-------
`(:Entity)-[:RELATES_TO]->(:Entity)` relationships carry an evidence/provenance list:
  - r.source_chunk_ids: list[str]

During ingest this list is append-only (new evidences are merged). When chunks are later deleted
(reindex / file delete), the `Chunk` nodes are removed but `RELATES_TO.source_chunk_ids` can keep
referencing deleted chunk ids, which can bias downstream graph scoring and break rebuild/analysis.

This script audits and optionally repairs the dangling references.
"""
import argparse
import os
import uuid
from pathlib import Path


def _repo_root() -> Path:
    # scripts/hipporag/maintenance/* -> repo root is three levels up
    return Path(__file__).resolve().parents[3]


def _strip_optional_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv(dotenv_path: Path, *, override: bool = False) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = _strip_optional_quotes(value)
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def _parse_uuid(value: str) -> uuid.UUID:
    token = (value or "").strip()
    if not token:
        raise ValueError("empty uuid")
    if len(token) == 32:
        return uuid.UUID(hex=token)
    return uuid.UUID(token)


def _print_counts(session, *, owner_id: str) -> None:  # noqa: ANN001
    q_total_refs = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.owner_id=$owner
    UNWIND coalesce(r.source_chunk_ids, []) AS cid
    RETURN count(cid) AS refs_total, count(DISTINCT cid) AS refs_distinct
    """
    rec = session.run(q_total_refs, owner=owner_id).single() or {}
    total_refs = int(rec.get("refs_total") or 0)
    distinct_refs = int(rec.get("refs_distinct") or 0)

    q_missing_distinct = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.owner_id=$owner
    UNWIND coalesce(r.source_chunk_ids, []) AS cid
    WITH DISTINCT cid
    WHERE NOT EXISTS { MATCH (:Chunk {chunk_id: cid, owner_id:$owner}) }
    RETURN count(cid) AS missing_distinct
    """
    rec2 = session.run(q_missing_distinct, owner=owner_id).single() or {}
    missing_distinct = int(rec2.get("missing_distinct") or 0)

    q_missing = """
    MATCH ()-[r:RELATES_TO]->()
    WHERE r.owner_id=$owner
    WITH r,
         [cid IN coalesce(r.source_chunk_ids, []) WHERE NOT EXISTS { MATCH (:Chunk {chunk_id: cid, owner_id:$owner}) }] AS missing
    RETURN count(r) AS rels_total,
           sum(CASE WHEN size(missing) > 0 THEN 1 ELSE 0 END) AS rels_with_missing,
           sum(size(missing)) AS missing_refs_total
    """
    rec3 = session.run(q_missing, owner=owner_id).single() or {}
    rels_total = int(rec3.get("rels_total") or 0)
    rels_with_missing = int(rec3.get("rels_with_missing") or 0)
    missing_refs_total = int(rec3.get("missing_refs_total") or 0)

    ratio = (missing_refs_total / total_refs) if total_refs else 0.0
    print(f"[audit] owner_id={owner_id}")
    print(f"[audit] relates_to_edges={rels_total}")
    print(f"[audit] source_chunk_ids refs_total={total_refs} distinct={distinct_refs}")
    print(f"[audit] dangling refs_total={missing_refs_total} distinct={missing_distinct} ratio={ratio:.3f}")
    if rels_total:
        print(f"[audit] rels_with_dangling={rels_with_missing}/{rels_total} ({(rels_with_missing/rels_total):.3f})")


def main(argv: list[str]) -> int:
    repo_root = _repo_root()
    default_env = repo_root / ".env"

    ap = argparse.ArgumentParser(description="Repair Neo4j RELATES_TO.source_chunk_ids dangling provenance (dry-run by default).")
    ap.add_argument("--owner-id", required=True, type=_parse_uuid)
    ap.add_argument("--dotenv", default=str(default_env))
    ap.add_argument("--apply", action="store_true", help="Apply repair in Neo4j. Default: audit only.")
    ap.add_argument("--batch-size", type=int, default=5000)
    args = ap.parse_args(argv)

    _load_dotenv(Path(args.dotenv), override=False)

    from neo4j import GraphDatabase

    owner_id = str(args.owner_id)
    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "")
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")

    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pass))
    with driver.session(database=neo4j_db) as session:
        _print_counts(session, owner_id=owner_id)

        if not bool(args.apply):
            print("[dry-run] no changes applied. Pass --apply to write updates.")
            return 0

        q_ids = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.owner_id=$owner
          AND ANY(cid IN coalesce(r.source_chunk_ids, []) WHERE NOT EXISTS { MATCH (:Chunk {chunk_id: cid, owner_id:$owner}) })
        RETURN id(r) AS rid
        """
        rel_ids = [int(r["rid"]) for r in session.run(q_ids, owner=owner_id)]
        print(f"[apply] relationships_to_repair={len(rel_ids)}")
        if not rel_ids:
            print("[apply] nothing to do.")
            return 0

        batch_size = max(1, int(args.batch_size))
        total_repaired = 0
        total_removed = 0
        total_now_empty = 0

        q_fix = """
        UNWIND $rids AS rid
        MATCH ()-[r:RELATES_TO]->() WHERE id(r)=rid AND r.owner_id=$owner
        WITH r,
             coalesce(r.source_chunk_ids, []) AS before,
             [cid IN coalesce(r.source_chunk_ids, []) WHERE EXISTS { MATCH (:Chunk {chunk_id: cid, owner_id:$owner}) }] AS after
        WITH r, before, after, (size(before) - size(after)) AS removed
        SET r.source_chunk_ids = after,
            r.source_chunk_ids_cleaned_at = datetime(),
            r.source_chunk_ids_removed_dangling = coalesce(r.source_chunk_ids_removed_dangling, 0) + removed
        RETURN count(r) AS repaired, sum(removed) AS removed_total, sum(CASE WHEN size(after)=0 THEN 1 ELSE 0 END) AS now_empty
        """

        for i in range(0, len(rel_ids), batch_size):
            batch = rel_ids[i : i + batch_size]
            rec = session.run(q_fix, owner=owner_id, rids=batch).single() or {}
            total_repaired += int(rec.get("repaired") or 0)
            total_removed += int(rec.get("removed_total") or 0)
            total_now_empty += int(rec.get("now_empty") or 0)
            print(f"[apply] batch={i//batch_size+1} repaired={total_repaired} removed={total_removed} now_empty={total_now_empty}")

        print(f"[apply] done: repaired={total_repaired} removed={total_removed} now_empty={total_now_empty}")
        _print_counts(session, owner_id=owner_id)
        return 0


if __name__ == "__main__":
    raise SystemExit(main(list(os.sys.argv[1:])))

