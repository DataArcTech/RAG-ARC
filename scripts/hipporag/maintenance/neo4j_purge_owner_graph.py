#!/usr/bin/env python3
"""
Owner-scoped purge for Neo4j HippoRAG graph data (dry-run by default).

Use case
--------
When doing a "full rebuild" for a tenant/owner, remove all owner-scoped nodes/edges from Neo4j so:
- stale `RELATES_TO.source_chunk_ids` provenance cannot accumulate across rebuilds
- chunks/entities/schema-layer nodes do not duplicate or drift across runs

Safety
------
- default is dry-run (prints counts only)
- `--apply` actually deletes
"""

from __future__ import annotations

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


def _count(session, query: str, **params):  # noqa: ANN001
    rec = session.run(query, **params).single() or {}
    return int(rec.get("n") or 0)


def _print_stats(session, owner_id: str) -> None:  # noqa: ANN001
    print(f"[stats] owner_id={owner_id}")
    print("[stats] nodes")
    print("  Chunk:", _count(session, "MATCH (n:Chunk {owner_id:$owner}) RETURN count(n) AS n", owner=owner_id))
    print("  Entity:", _count(session, "MATCH (n:Entity {owner_id:$owner}) RETURN count(n) AS n", owner=owner_id))
    print("  EntityAlias:", _count(session, "MATCH (n:EntityAlias {owner_id:$owner}) RETURN count(n) AS n", owner=owner_id))
    print("  SchemaNode:", _count(session, "MATCH (n:SchemaNode {owner_id:$owner}) RETURN count(n) AS n", owner=owner_id))
    print("  SDFEvent:", _count(session, "MATCH (n:SDFEvent {owner_id:$owner}) RETURN count(n) AS n", owner=owner_id))
    print("  KGIngestMeta:", _count(session, "MATCH (n:KGIngestMeta {owner_id:$owner}) RETURN count(n) AS n", owner=owner_id))
    print("[stats] relationships")
    print(
        "  MENTIONS:",
        _count(
            session,
            "MATCH (:Chunk {owner_id:$owner})-[r:MENTIONS]->(:Entity {owner_id:$owner}) RETURN count(r) AS n",
            owner=owner_id,
        ),
    )
    print(
        "  RELATES_TO:",
        _count(
            session,
            "MATCH (:Entity {owner_id:$owner})-[r:RELATES_TO]->(:Entity {owner_id:$owner}) RETURN count(r) AS n",
            owner=owner_id,
        ),
    )
    print(
        "  RELATES_TO(owner_id property):",
        _count(session, "MATCH ()-[r:RELATES_TO]->() WHERE r.owner_id=$owner RETURN count(r) AS n", owner=owner_id),
    )


def main(argv: list[str]) -> int:
    repo_root = _repo_root()
    default_env = repo_root / ".env"

    ap = argparse.ArgumentParser(description="Owner-scoped purge of Neo4j HippoRAG graph data (dry-run by default).")
    ap.add_argument("--owner-id", required=True, type=_parse_uuid)
    ap.add_argument("--dotenv", default=str(default_env))
    ap.add_argument("--apply", action="store_true", help="Actually delete data. Default: dry-run stats only.")
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
        _print_stats(session, owner_id)
        if not args.apply:
            print("[dry-run] no changes applied. Pass --apply to delete.")
            return 0

        session.run("MATCH ()-[r:RELATES_TO]->() WHERE r.owner_id=$owner DELETE r", owner=owner_id)
        session.run("MATCH (n:Chunk {owner_id:$owner}) DETACH DELETE n", owner=owner_id)
        session.run("MATCH (n:Entity {owner_id:$owner}) DETACH DELETE n", owner=owner_id)
        session.run("MATCH (n:EntityAlias {owner_id:$owner}) DETACH DELETE n", owner=owner_id)
        session.run("MATCH (n:SchemaNode {owner_id:$owner}) DETACH DELETE n", owner=owner_id)
        session.run("MATCH (n:SDFEvent {owner_id:$owner}) DETACH DELETE n", owner=owner_id)
        session.run("MATCH (n:KGIngestMeta {owner_id:$owner}) DETACH DELETE n", owner=owner_id)

        print("[apply] purge completed; post-stats:")
        _print_stats(session, owner_id)
        return 0


if __name__ == "__main__":
    raise SystemExit(main(list(os.sys.argv[1:])))

