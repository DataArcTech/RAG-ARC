#!/usr/bin/env python3
"""
Backfill PageIndex-aligned Section nodes + edges from existing Chunk metadata (Neo4j).

Default: apply changes (test environments). Use --dry-run to only print counts.
"""
import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config.core.deepsearch import tool_defaults  # noqa: E402
from core.deepsearch.utils.node_types import normalize_node_type  # noqa: E402


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


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _tree_node_summary(metadata: Dict[str, Any], content: str) -> str | None:
    summary = ""
    index_text = metadata.get("index_text")
    if isinstance(index_text, str) and index_text.strip():
        summary = index_text.strip()
    if not summary:
        caption = metadata.get("table_caption")
        if isinstance(caption, str) and caption.strip():
            summary = caption.strip()
    if not summary:
        alts = metadata.get("image_alts")
        if isinstance(alts, list):
            cleaned = [str(a).strip() for a in alts if str(a or "").strip()]
            if cleaned:
                summary = "; ".join(cleaned)
    if not summary:
        summary = str(content or "").strip()
    if not summary:
        return None
    max_chars = int(getattr(tool_defaults, "SECTION_SEARCH_SUMMARY_PREVIEW_CHARS", 160))
    max_chars = max(1, max_chars)
    if len(summary) > max_chars:
        return summary[:max_chars].rstrip() + "..."
    return summary


def _collect_sections(
    rows: List[Dict[str, Any]],
    *,
    owner_id: str,
    delimiter: str,
    section_data_by_key: Dict[Tuple[str, str], Dict[str, Any]],
    section_chunk_links: List[Dict[str, Any]],
    tree_node_data_by_key: Dict[Tuple[str, str], Dict[str, Any]],
    tree_node_section_links: List[Dict[str, Any]],
    tree_node_chunk_links: List[Dict[str, Any]],
    tree_node_parent_links: List[Dict[str, Any]],
    tree_node_section_link_keys: set[Tuple[str, str, str]],
    tree_node_chunk_link_keys: set[Tuple[str, str, str]],
    tree_node_parent_link_keys: set[Tuple[str, str, str]],
) -> None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id") or "").strip()
        source_file_id = str(row.get("source_file_id") or "").strip()
        if not chunk_id or not source_file_id:
            continue
        meta = _parse_metadata(row.get("metadata"))
        content = str(row.get("content") or "")
        section_id = str(meta.get("section_id") or "").strip()
        if not section_id:
            continue

        section_key = (owner_id, section_id)
        section_path = str(meta.get("section_path") or "").strip()
        section_title = str(meta.get("section_title") or "").strip()
        if not section_title and section_path:
            section_title = section_path.split(delimiter)[-1].strip() if delimiter in section_path else section_path
        section_level = _coerce_int(meta.get("section_level"))
        section_parent_id = str(meta.get("section_parent_id") or "").strip() or None
        page_start = _coerce_int(meta.get("section_page_start"))
        if page_start is None:
            page_start = _coerce_int(meta.get("page_start"))
        page_end = _coerce_int(meta.get("section_page_end"))
        if page_end is None:
            page_end = _coerce_int(meta.get("page_end"))

        existing = section_data_by_key.get(section_key)
        if existing is None:
            section_data_by_key[section_key] = {
                "section_id": section_id,
                "owner_id": owner_id,
                "source_file_id": source_file_id,
                "section_path": section_path or None,
                "section_title": section_title or None,
                "section_level": section_level,
                "section_parent_id": section_parent_id,
                "page_start": page_start,
                "page_end": page_end,
            }
        else:
            if not existing.get("section_path") and section_path:
                existing["section_path"] = section_path
            if not existing.get("section_title") and section_title:
                existing["section_title"] = section_title
            if existing.get("section_level") is None and section_level is not None:
                existing["section_level"] = section_level
            if not existing.get("section_parent_id") and section_parent_id:
                existing["section_parent_id"] = section_parent_id
            if page_start is not None:
                prior = existing.get("page_start")
                existing["page_start"] = page_start if prior is None else min(int(prior), page_start)
            if page_end is not None:
                prior = existing.get("page_end")
                existing["page_end"] = page_end if prior is None else max(int(prior), page_end)

        section_chunk_links.append({"section_id": section_id, "chunk_id": chunk_id, "owner_id": owner_id})

        tree_node_id = str(meta.get("semantic_unit_id") or "").strip() or chunk_id
        semantic_unit_type = str(meta.get("semantic_unit_type") or "").strip() or "text"
        node_type = normalize_node_type(semantic_unit_type)
        page_start = _coerce_int(meta.get("page_start"))
        page_end = _coerce_int(meta.get("page_end"))
        summary = _tree_node_summary(meta, content)
        token_count = _coerce_int(meta.get("token_count"))
        resource_urls = meta.get("image_urls") if isinstance(meta.get("image_urls"), list) else None
        cleaned_urls = []
        if isinstance(resource_urls, list):
            cleaned_urls = [str(u).strip() for u in resource_urls if str(u or "").strip()]
        if tree_node_id:
            tree_key = (owner_id, tree_node_id)
            existing = tree_node_data_by_key.get(tree_key)
            if existing is None:
                tree_node_data_by_key[tree_key] = {
                    "node_id": tree_node_id,
                    "owner_id": owner_id,
                    "source_file_id": source_file_id,
                    "node_type": node_type,
                    "semantic_unit_type": semantic_unit_type,
                    "section_id": section_id or None,
                    "section_path": section_path or None,
                    "page_start": page_start,
                    "page_end": page_end,
                    "summary": summary,
                    "resource_urls": cleaned_urls or None,
                    "resource_paths": cleaned_urls or None,
                    "token_count": token_count,
                }
            else:
                if not existing.get("node_type") or (
                    existing.get("node_type") == tool_defaults.SECTION_NODE_TYPE_DEFAULT
                    and node_type != tool_defaults.SECTION_NODE_TYPE_DEFAULT
                ):
                    existing["node_type"] = node_type
                if not existing.get("semantic_unit_type") and semantic_unit_type:
                    existing["semantic_unit_type"] = semantic_unit_type
                if not existing.get("section_id") and section_id:
                    existing["section_id"] = section_id
                if not existing.get("section_path") and section_path:
                    existing["section_path"] = section_path
                if page_start is not None:
                    prior = existing.get("page_start")
                    existing["page_start"] = page_start if prior is None else min(int(prior), page_start)
                if page_end is not None:
                    prior = existing.get("page_end")
                    existing["page_end"] = page_end if prior is None else max(int(prior), page_end)
                if summary and not existing.get("summary"):
                    existing["summary"] = summary
                if cleaned_urls and not existing.get("resource_urls"):
                    existing["resource_urls"] = cleaned_urls
                if cleaned_urls and not existing.get("resource_paths"):
                    existing["resource_paths"] = cleaned_urls
                if token_count is not None:
                    existing["token_count"] = int(existing.get("token_count") or 0) + token_count

            chunk_link_key = (owner_id, tree_node_id, chunk_id)
            if chunk_link_key not in tree_node_chunk_link_keys:
                tree_node_chunk_link_keys.add(chunk_link_key)
                tree_node_chunk_links.append({"node_id": tree_node_id, "chunk_id": chunk_id, "owner_id": owner_id})

            section_link_key = (owner_id, section_id, tree_node_id)
            if section_link_key not in tree_node_section_link_keys:
                tree_node_section_link_keys.add(section_link_key)
                tree_node_section_links.append({"section_id": section_id, "node_id": tree_node_id, "owner_id": owner_id})

            parent_unit_id = str(meta.get("parent_unit_id") or "").strip()
            if parent_unit_id and parent_unit_id != tree_node_id:
                parent_link_key = (owner_id, parent_unit_id, tree_node_id)
                if parent_link_key not in tree_node_parent_link_keys:
                    tree_node_parent_link_keys.add(parent_link_key)
                    tree_node_parent_links.append(
                        {"parent_id": parent_unit_id, "node_id": tree_node_id, "owner_id": owner_id}
                    )


def _resolve_parents(section_data_by_key: Dict[Tuple[str, str], Dict[str, Any]], *, delimiter: str) -> List[Dict[str, Any]]:
    path_index: Dict[Tuple[str, str, str], str] = {}
    for record in section_data_by_key.values():
        owner_id = str(record.get("owner_id") or "").strip()
        source_file_id = str(record.get("source_file_id") or "").strip()
        section_path = str(record.get("section_path") or "").strip()
        section_id = str(record.get("section_id") or "").strip()
        if owner_id and source_file_id and section_path and section_id:
            path_index[(owner_id, source_file_id, section_path)] = section_id

    for record in section_data_by_key.values():
        if record.get("section_parent_id"):
            continue
        section_path = str(record.get("section_path") or "").strip()
        if not section_path or delimiter not in section_path:
            continue
        parent_path = delimiter.join([seg for seg in section_path.split(delimiter)[:-1] if seg])
        if not parent_path:
            continue
        owner_id = str(record.get("owner_id") or "").strip()
        source_file_id = str(record.get("source_file_id") or "").strip()
        parent_id = path_index.get((owner_id, source_file_id, parent_path))
        if parent_id:
            record["section_parent_id"] = parent_id

    return [
        {"owner_id": r.get("owner_id"), "section_id": r.get("section_id"), "parent_id": r.get("section_parent_id")}
        for r in section_data_by_key.values()
        if r.get("section_parent_id")
    ]


def main(argv: List[str]) -> int:
    repo_root = _repo_root()
    default_env = repo_root / ".env"

    ap = argparse.ArgumentParser(description="Backfill PageIndex Section nodes/edges from Chunk metadata (dry-run by default).")
    ap.add_argument("--owner-id", required=True, type=_parse_uuid)
    ap.add_argument(
        "--source-file-ids",
        default="",
        help="Optional comma-separated source_file_id list to limit the backfill scope.",
    )
    ap.add_argument("--dotenv", default=str(default_env))
    ap.add_argument("--dry-run", action="store_true", help="Only print counts; do not write to Neo4j.")
    ap.add_argument("--batch-size", type=int, default=2000)
    args = ap.parse_args(argv)

    _load_dotenv(Path(args.dotenv), override=False)

    from neo4j import GraphDatabase

    owner_id = str(args.owner_id)
    delimiter = os.getenv("SECTION_PATH_DELIMITER", " > ")
    neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "")
    neo4j_db = os.getenv("NEO4J_DATABASE", "neo4j")

    section_data_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    section_chunk_links: List[Dict[str, Any]] = []
    tree_node_data_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    tree_node_section_links: List[Dict[str, Any]] = []
    tree_node_chunk_links: List[Dict[str, Any]] = []
    tree_node_parent_links: List[Dict[str, Any]] = []
    tree_node_section_link_keys: set[Tuple[str, str, str]] = set()
    tree_node_chunk_link_keys: set[Tuple[str, str, str]] = set()
    tree_node_parent_link_keys: set[Tuple[str, str, str]] = set()

    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_pass))
    with driver.session(database=neo4j_db) as session:
        batch_size = max(1, int(args.batch_size))
        skip = 0
        file_ids = [fid.strip() for fid in str(args.source_file_ids or "").split(",") if fid.strip()]
        file_filter = " AND c.source_file_id IN $file_ids" if file_ids else ""
        while True:
            query = """
            MATCH (c:Chunk {owner_id: $owner})
            WHERE 1 = 1
            """ + file_filter + """
            RETURN c.chunk_id AS chunk_id, c.metadata AS metadata, c.source_file_id AS source_file_id, c.content AS content
            SKIP $skip LIMIT $limit
            """
            params = {"owner": owner_id, "skip": skip, "limit": batch_size}
            if file_ids:
                params["file_ids"] = file_ids
            rows = [dict(r) for r in session.run(query, **params)]
            if not rows:
                break
            _collect_sections(
                rows,
                owner_id=owner_id,
                delimiter=delimiter,
                section_data_by_key=section_data_by_key,
                section_chunk_links=section_chunk_links,
                tree_node_data_by_key=tree_node_data_by_key,
                tree_node_section_links=tree_node_section_links,
                tree_node_chunk_links=tree_node_chunk_links,
                tree_node_parent_links=tree_node_parent_links,
                tree_node_section_link_keys=tree_node_section_link_keys,
                tree_node_chunk_link_keys=tree_node_chunk_link_keys,
                tree_node_parent_link_keys=tree_node_parent_link_keys,
            )
            skip += batch_size

        parent_links = _resolve_parents(section_data_by_key, delimiter=delimiter)

        print(f"[scan] owner_id={owner_id}")
        if file_ids:
            print(f"[scan] source_file_ids={len(file_ids)}")
        print(f"[scan] sections={len(section_data_by_key)}")
        print(f"[scan] section_chunk_links={len(section_chunk_links)}")
        print(f"[scan] parent_links={len(parent_links)}")
        print(f"[scan] tree_nodes={len(tree_node_data_by_key)}")
        print(f"[scan] tree_node_section_links={len(tree_node_section_links)}")
        print(f"[scan] tree_node_chunk_links={len(tree_node_chunk_links)}")
        print(f"[scan] tree_node_parent_links={len(tree_node_parent_links)}")

        if args.dry_run:
            print("[dry-run] no changes applied. Re-run without --dry-run to write updates.")
            return 0

        section_query = """
        UNWIND $sections AS section
        MERGE (s:Section {section_id: section.section_id, owner_id: section.owner_id})
        SET s.source_file_id = COALESCE(section.source_file_id, s.source_file_id),
            s.section_path = COALESCE(section.section_path, s.section_path),
            s.section_title = COALESCE(section.section_title, s.section_title),
            s.section_level = COALESCE(section.section_level, s.section_level),
            s.section_parent_id = COALESCE(section.section_parent_id, s.section_parent_id),
            s.page_start = COALESCE(section.page_start, s.page_start),
            s.page_end = COALESCE(section.page_end, s.page_end),
            s.updated_at = datetime(),
            s.created_at = COALESCE(s.created_at, datetime())
        """
        session.run(section_query, sections=list(section_data_by_key.values()))

        if parent_links:
            parent_query = """
            UNWIND $links AS link
            MATCH (c:Section {section_id: link.section_id, owner_id: link.owner_id})
            MATCH (p:Section {section_id: link.parent_id, owner_id: link.owner_id})
            MERGE (p)-[r:PARENT_OF {section_id: link.section_id, parent_id: link.parent_id}]->(c)
            SET r.updated_at = datetime(),
                r.created_at = COALESCE(r.created_at, datetime())
            """
            session.run(parent_query, links=parent_links)

        if section_chunk_links:
            chunk_link_query = """
            UNWIND $links AS link
            MATCH (s:Section {section_id: link.section_id, owner_id: link.owner_id})
            MATCH (c:Chunk {chunk_id: link.chunk_id, owner_id: link.owner_id})
            MERGE (s)-[r:HAS_CHUNK {section_id: link.section_id, chunk_id: link.chunk_id}]->(c)
            SET r.owner_id = link.owner_id,
                r.updated_at = datetime(),
                r.created_at = COALESCE(r.created_at, datetime())
            """
            session.run(chunk_link_query, links=section_chunk_links)

        if tree_node_data_by_key:
            tree_node_query = """
            UNWIND $nodes AS node
            MERGE (t:TreeNode {node_id: node.node_id, owner_id: node.owner_id})
            SET t.source_file_id = COALESCE(node.source_file_id, t.source_file_id),
                t.section_id = COALESCE(node.section_id, t.section_id),
                t.section_path = COALESCE(node.section_path, t.section_path),
                t.node_type = COALESCE(node.node_type, t.node_type),
                t.semantic_unit_type = COALESCE(node.semantic_unit_type, t.semantic_unit_type),
                t.page_start = CASE WHEN node.page_start IS NULL THEN t.page_start ELSE node.page_start END,
                t.page_end = CASE WHEN node.page_end IS NULL THEN t.page_end ELSE node.page_end END,
                t.summary = CASE WHEN node.summary IS NULL OR node.summary = '' THEN t.summary ELSE node.summary END,
                t.resource_urls = CASE
                    WHEN node.resource_urls IS NULL OR size(node.resource_urls) = 0 THEN t.resource_urls
                    ELSE node.resource_urls
                END,
                t.resource_paths = CASE
                    WHEN node.resource_paths IS NULL OR size(node.resource_paths) = 0 THEN t.resource_paths
                    ELSE node.resource_paths
                END,
                t.token_count = COALESCE(node.token_count, t.token_count),
                t.updated_at = datetime(),
                t.created_at = COALESCE(t.created_at, datetime())
            """
            session.run(tree_node_query, nodes=list(tree_node_data_by_key.values()))

        if tree_node_section_links:
            section_tree_link_query = """
            UNWIND $links AS link
            MATCH (s:Section {section_id: link.section_id, owner_id: link.owner_id})
            MATCH (t:TreeNode {node_id: link.node_id, owner_id: link.owner_id})
            MERGE (s)-[r:HAS_CHILD {section_id: link.section_id, node_id: link.node_id}]->(t)
            SET r.owner_id = link.owner_id,
                r.updated_at = datetime(),
                r.created_at = COALESCE(r.created_at, datetime())
            """
            session.run(section_tree_link_query, links=tree_node_section_links)

        if tree_node_parent_links:
            node_parent_query = """
            UNWIND $links AS link
            MATCH (p:TreeNode {node_id: link.parent_id, owner_id: link.owner_id})
            MATCH (c:TreeNode {node_id: link.node_id, owner_id: link.owner_id})
            MERGE (p)-[r:HAS_CHILD {parent_id: link.parent_id, node_id: link.node_id}]->(c)
            SET r.owner_id = link.owner_id,
                r.updated_at = datetime(),
                r.created_at = COALESCE(r.created_at, datetime())
            """
            session.run(node_parent_query, links=tree_node_parent_links)

        if tree_node_chunk_links:
            tree_node_chunk_query = """
            UNWIND $links AS link
            MATCH (t:TreeNode {node_id: link.node_id, owner_id: link.owner_id})
            MATCH (c:Chunk {chunk_id: link.chunk_id, owner_id: link.owner_id})
            MERGE (t)-[r:HAS_CHUNK {node_id: link.node_id, chunk_id: link.chunk_id}]->(c)
            SET r.owner_id = link.owner_id,
                r.updated_at = datetime(),
                r.created_at = COALESCE(r.created_at, datetime())
            """
            session.run(tree_node_chunk_query, links=tree_node_chunk_links)

    print("[apply] backfill completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
