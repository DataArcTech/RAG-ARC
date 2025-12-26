#!/usr/bin/env python3
"""Diagnose whether the Neo4j-backed graph store contains chunks/entities usable by DeepSearch.

This script is meant for quick "why evidence_count=0" triage:
- Is Neo4j reachable?
- What labels/indexes exist?
- Are there any chunk-like nodes and do they contain content-like properties?
"""
import argparse
import os
from typing import Any, Dict, List, Tuple

from neo4j import GraphDatabase


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _run(session, cypher: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    result = session.run(cypher, params or {})
    return [dict(record.data()) for record in result]


def _top_label_candidates(labels: List[str]) -> List[str]:
    scored: List[Tuple[int, str]] = []
    for label in labels:
        token = str(label or "").strip()
        if not token:
            continue
        score = 0
        lower = token.lower()
        if "chunk" in lower:
            score += 10
        if "doc" in lower:
            score += 6
        if "entity" in lower:
            score += 5
        if "text" in lower:
            score += 3
        scored.append((score, token))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered = [label for _, label in scored]
    return ordered[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=_env("NEO4J_URL", "bolt://localhost:7687"))
    parser.add_argument("--username", default=_env("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--password", default=_env("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=_env("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--sample", default="", help="Optional sample substring to search in content-like fields.")
    args = parser.parse_args()

    if not args.password:
        raise SystemExit("NEO4J_PASSWORD is empty; cannot connect.")

    driver = GraphDatabase.driver(args.url, auth=(args.username, args.password))
    try:
        with driver.session(database=args.database) as session:
            print(f"[neo4j] url={args.url} database={args.database}")
            info = _run(session, "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition")
            if info:
                print("[neo4j] component:", info[0])

            labels_raw = _run(session, "CALL db.labels() YIELD label RETURN label ORDER BY label")
            labels = [row["label"] for row in labels_raw if row.get("label")]
            print(f"[neo4j] labels={len(labels)}")
            for label in _top_label_candidates(labels):
                count_rows = _run(session, f"MATCH (n:`{label}`) RETURN count(n) AS n")
                n = int((count_rows[0] or {}).get("n") or 0) if count_rows else 0
                print(f"- label `{label}` count={n}")
                if n <= 0:
                    continue
                keys_rows = _run(session, f"MATCH (n:`{label}`) RETURN keys(n) AS keys LIMIT 1")
                keys = (keys_rows[0] or {}).get("keys") if keys_rows else None
                print(f"  keys={keys}")

            indexes: list[dict] = []
            try:
                indexes = _run(
                    session,
                    "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, state "
                    "RETURN name, type, entityType, labelsOrTypes, properties, state ORDER BY name",
                )
            except Exception:
                try:
                    indexes = _run(
                        session,
                        "CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, state "
                        "RETURN name, type, entityType, labelsOrTypes, properties, state ORDER BY name",
                    )
                except Exception:
                    indexes = []
            print(f"[neo4j] indexes={len(indexes)}")
            for row in indexes[:30]:
                print("-", row)

            if args.sample:
                sample = str(args.sample).strip()
                if sample:
                    print(f"[sample] searching for substring: {sample!r}")
                    # Heuristic scan: for each candidate label, try the most common content field names.
                    for label in _top_label_candidates(labels):
                        for field in ("content", "text", "chunk", "body"):
                            try:
                                rows = _run(
                                    session,
                                    f"MATCH (n:`{label}`) WHERE n.{field} CONTAINS $q "
                                    f"RETURN id(n) AS id, substring(n.{field}, 0, 240) AS preview LIMIT 3",
                                    {"q": sample},
                                )
                            except Exception:
                                rows = []
                            if rows:
                                print(f"[sample] hit label={label} field={field} rows={len(rows)}")
                                for r in rows:
                                    print(" ", r)
                                break

    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
