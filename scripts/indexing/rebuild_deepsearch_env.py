#!/usr/bin/env python3
"""Rebuild DeepSearch benchmark data for test + bench owners in one run.

Goals:
- Reindex test-owner files from existing parsed artifacts.
- Rebuild bench-owner files by reusing parsed blobs from source owner and indexing them.
- Verify Postgres/Neo4j state is healthy for subsequent DeepSearch benchmarks.
"""
import argparse
import asyncio
import json
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text

from cli.bootstrap import initialize
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from core.file_management.storage.file import FileValidationError
from encapsulation.data_model.orm_models import User
from framework.register import Register


DEFAULT_TEST_OWNER_ID = "818eaeef-bed4-4b58-8d4e-e86addc8030e"
DEFAULT_BENCH_OWNER_ID = "212ee819-7e22-4f5d-94bd-30146e514f60"
DEFAULT_TEST_FILE_IDS = [
    "fd16f299-32fe-4390-8b7d-4262e1786bf2",
    "58950123-eae7-4554-91cd-8960bafce174",
]
DEFAULT_BENCH_PDFS = [
    "docs-proj/test_pdfs/「星鑽」儲蓄壽險計劃 II-產品資料冊.pdf",
    "docs-proj/test_pdfs/星鑽儲蓄壽險計劃II-小册子.pdf",
]
DEFAULT_REPORT_PATH = "local/bench/rebuild_deepsearch_env_report.json"


def _parse_uuid(value: str, *, field_name: str) -> uuid.UUID:
    token = str(value or "").strip()
    if not token:
        raise ValueError(f"{field_name} must not be empty")
    try:
        return uuid.UUID(token)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{field_name} must be a valid UUID: {value}") from exc


def _normalize_path_key(value: str) -> str:
    return str(value or "").replace("\\", "/").strip()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_knowledge_config_path(raw: str) -> Path:
    token = str(raw or "").strip() or "config/json_configs/knowledge.json"
    path = Path(token)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _build_fast_knowledge_config(base_config_path: Path) -> Path:
    payload = json.loads(base_config_path.read_text(encoding="utf-8"))
    indexers = payload.get("index_manager_config", {}).get("indexer_configs", [])
    for indexer in indexers:
        if not isinstance(indexer, dict):
            continue
        if str(indexer.get("type") or "") != "pruned_hipporag_neo4j_indexer":
            continue
        extractor_cfg = indexer.setdefault("extractor_config", {})
        # Keep graph ingestion path active, but skip LLM extraction to avoid unstable structured-output calls.
        extractor_cfg["extract_chunk_roles"] = ["__skip__"]
        extractor_cfg["error_policy"] = "attach"

    out_path = Path("local/bench/knowledge_rebuild_fast.json")
    _ensure_parent(out_path)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path.resolve()


@dataclass
class FileVerifyResult:
    file_id: str
    owner_id: str
    filename: str
    status: str
    chunk_count: int
    chunk_with_page_start: int
    chunk_with_page_end: int


@dataclass
class RebuildResult:
    test_owner_id: str
    bench_owner_id: str
    source_owner_id: str
    knowledge_config_path: str
    parser_parse_mode: str
    fast_rebuild_mode: bool
    reset_bench_owner: bool
    test_reindex_results: list[dict[str, Any]]
    bench_rebuild_results: list[dict[str, Any]]
    verification: list[dict[str, Any]]
    generated_at: str


def _fetch_source_rows(source_owner_id: uuid.UUID) -> list[dict[str, Any]]:
    db = PostgreSQLConfig().build()
    with db.engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT fm.file_id,
                       fm.filename,
                       pcm.parsed_content_id,
                       pcm.parser_type,
                       pcm.md_path,
                       pcm.output_dir,
                       pcm.created_at
                FROM file_metadata fm
                JOIN LATERAL (
                    SELECT parsed_content_id,
                           parser_type,
                           md_path,
                           output_dir,
                           created_at
                    FROM parsed_content_metadata
                    WHERE source_file_id = fm.file_id
                    ORDER BY created_at DESC
                    LIMIT 1
                ) pcm ON TRUE
                WHERE fm.owner_id = :owner_id
                  AND fm.status != 'DELETED'
                ORDER BY fm.updated_at DESC
                """
            ),
            {"owner_id": source_owner_id},
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def _match_source_row(*, pdf_path: Path, source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = _normalize_path_key(pdf_path.as_posix())
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for row in source_rows:
        filename = _normalize_path_key(str(row.get("filename") or ""))
        if not filename:
            continue
        score = 0
        if filename == target:
            score = 4
        elif filename.endswith("/" + target):
            score = 3
        elif Path(filename).name == pdf_path.name:
            score = 2
        elif pdf_path.stem in filename:
            score = 1
        if score > 0:
            created = str(row.get("created_at") or "")
            candidates.append((score, created, row))

    if not candidates:
        raise RuntimeError(f"No parsed source match found for bench PDF: {pdf_path}")

    # Prefer strongest path match; tie-break by parsed created_at (newer first).
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _ensure_user(account: Any, owner_id: uuid.UUID, *, username_prefix: str) -> None:
    try:
        existing = account.get_user_by_id(owner_id)
    except Exception:  # noqa: BLE001
        existing = None
    if existing:
        return

    username = f"{username_prefix}_{owner_id.hex[:8]}"
    hashed = account.get_password_hash("deepsearch-rebuild")
    user = User(
        id=owner_id,
        user_name=username,
        hashed_password=hashed,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    account.user_storage.metadata_store.store_user(user)


async def _reset_owner_files(knowledge: Any, owner_id: uuid.UUID) -> list[str]:
    files = knowledge.list_user_files(user_id=owner_id, status=None, limit=None, offset=None)
    deleted: list[str] = []
    for file_meta in files:
        fid = str(getattr(file_meta, "file_id", "") or "").strip()
        if not fid:
            continue
        await knowledge.mark_file_deleted_cli(fid, owner_id)
        deleted.append(fid)
    return deleted


def _latest_parsed_id_for_file(knowledge: Any, *, file_id: str) -> str:
    parsed_list = knowledge.file_index.parsed_content_storage.metadata_store.list_parsed_content_metadata(
        source_file_id=file_id,
        limit=1,
    )
    if not parsed_list:
        raise RuntimeError(f"No parsed content found for file_id={file_id}")
    parsed_id = str(getattr(parsed_list[0], "parsed_content_id", "") or "").strip()
    if not parsed_id:
        raise RuntimeError(f"Invalid parsed_content_id for file_id={file_id}")
    return parsed_id


async def _reindex_file_from_parsed(knowledge: Any, *, file_id: str, parsed_content_id: str) -> dict[str, Any]:
    result = await knowledge.file_index.index_file(
        file_id,
        parsed_content_id=parsed_content_id,
        reuse_parsed=True,
    )
    return {
        "file_id": file_id,
        "parsed_content_id": parsed_content_id,
        "success": bool(result.get("success")),
        "num_chunks": int((result.get("metadata") or {}).get("num_chunks") or 0),
        "error_message": result.get("error_message"),
    }


def _upload_or_reuse_file(knowledge: Any, *, owner_id: uuid.UUID, filename: str, file_data: bytes) -> tuple[str, bool]:
    try:
        file_id = knowledge.file_storage.upload_file(
            filename=filename,
            file_data=file_data,
            owner_id=owner_id,
            content_type="application/pdf",
        )
        return str(file_id), False
    except FileValidationError:
        duplicates = knowledge.file_storage.find_duplicate_file_ids(
            filename=filename,
            file_data=file_data,
            owner_id=owner_id,
        )
        if duplicates:
            return str(duplicates[0]), True
        raise


def _verify_files_in_postgres(file_ids: list[str]) -> dict[str, dict[str, Any]]:
    db = PostgreSQLConfig().build()
    output: dict[str, dict[str, Any]] = {}
    with db.engine.connect() as conn:
        for file_id in file_ids:
            row = conn.execute(
                text(
                    """
                    SELECT file_id, owner_id, filename, status
                    FROM file_metadata
                    WHERE file_id = :file_id
                    """
                ),
                {"file_id": file_id},
            ).fetchone()
            if row is None:
                output[file_id] = {
                    "file_id": file_id,
                    "owner_id": None,
                    "filename": None,
                    "status": "MISSING",
                }
            else:
                payload = dict(row._mapping)
                payload["owner_id"] = str(payload.get("owner_id"))
                output[file_id] = payload
    return output


def _verify_chunks_in_neo4j(file_ids: list[str]) -> dict[str, dict[str, int]]:
    from neo4j import GraphDatabase

    url = str(os.getenv("NEO4J_URL") or "").strip()
    user = str(os.getenv("NEO4J_USERNAME") or "").strip()
    password = str(os.getenv("NEO4J_PASSWORD") or "").strip()
    database = str(os.getenv("NEO4J_DATABASE") or "neo4j").strip()
    if not (url and user):
        raise RuntimeError("Missing NEO4J_URL/NEO4J_USERNAME in environment")

    driver = GraphDatabase.driver(url, auth=(user, password))
    try:
        out: dict[str, dict[str, int]] = {}
        with driver.session(database=database) as session:
            for file_id in file_ids:
                rec = session.run(
                    (
                        "MATCH (c:Chunk {source_file_id:$file_id}) "
                        "RETURN count(c) AS chunk_count, "
                        "count(c.page_start) AS chunk_with_page_start, "
                        "count(c.page_end) AS chunk_with_page_end"
                    ),
                    file_id=file_id,
                ).single()
                out[file_id] = {
                    "chunk_count": int(rec["chunk_count"]),
                    "chunk_with_page_start": int(rec["chunk_with_page_start"]),
                    "chunk_with_page_end": int(rec["chunk_with_page_end"]),
                }
        return out
    finally:
        driver.close()


async def _run(args: argparse.Namespace) -> RebuildResult:
    test_owner_id = _parse_uuid(args.test_owner_id, field_name="test_owner_id")
    bench_owner_id = _parse_uuid(args.bench_owner_id, field_name="bench_owner_id")
    source_owner_id = _parse_uuid(args.source_owner_id, field_name="source_owner_id")

    if args.fast_rebuild_mode:
        base_cfg = _resolve_knowledge_config_path(os.getenv("KNOWLEDGE_CONFIG_PATH", "config/json_configs/knowledge.json"))
        fast_cfg = _build_fast_knowledge_config(base_cfg)
        os.environ["KNOWLEDGE_CONFIG_PATH"] = str(fast_cfg)
        print(f"[rebuild] fast mode enabled, knowledge config -> {fast_cfg}")

    initialize(owner_id=str(test_owner_id))
    reg = Register()
    knowledge = reg.get_object("knowledge")
    account = reg.get_object("account")

    _ensure_user(account, test_owner_id, username_prefix="test_rebuild")
    _ensure_user(account, bench_owner_id, username_prefix="bench_rebuild")

    if args.reset_bench_owner:
        deleted = await _reset_owner_files(knowledge, bench_owner_id)
        print(f"[rebuild] bench owner reset deleted_files={len(deleted)}")

    test_results: list[dict[str, Any]] = []
    for file_id in args.test_file_id:
        parsed_content_id = _latest_parsed_id_for_file(knowledge, file_id=file_id)
        result = await _reindex_file_from_parsed(knowledge, file_id=file_id, parsed_content_id=parsed_content_id)
        test_results.append(result)
        print(f"[rebuild] test file reindexed file_id={file_id} success={result['success']} chunks={result['num_chunks']}")

    source_rows = _fetch_source_rows(source_owner_id)
    bench_results: list[dict[str, Any]] = []
    for raw_pdf in args.bench_pdf:
        pdf_path = Path(raw_pdf).expanduser().resolve()
        if not pdf_path.exists():
            raise RuntimeError(f"Bench PDF not found: {pdf_path}")

        source_row = _match_source_row(pdf_path=pdf_path, source_rows=source_rows)
        source_parsed_id = str(source_row.get("parsed_content_id") or "").strip()
        if not source_parsed_id:
            raise RuntimeError(f"Missing source parsed_content_id for matched source row: {source_row}")

        file_data = pdf_path.read_bytes()
        bench_filename = str(source_row.get("filename") or pdf_path.as_posix()).strip() or pdf_path.as_posix()
        bench_file_id, reused_existing = _upload_or_reuse_file(
            knowledge,
            owner_id=bench_owner_id,
            filename=bench_filename,
            file_data=file_data,
        )

        parsed_bytes = knowledge.file_index.parsed_content_storage.get_parsed_content(source_parsed_id)
        if not parsed_bytes:
            raise RuntimeError(f"Failed to load parsed blob from source parsed_content_id={source_parsed_id}")

        new_parsed_id = knowledge.file_index.parsed_content_storage.store_parsed_content(
            source_file_id=bench_file_id,
            parser_type=str(source_row.get("parser_type") or "mineru"),
            parsed_data=parsed_bytes,
            content_type="text/markdown",
            md_path=source_row.get("md_path"),
            output_dir=source_row.get("output_dir"),
        )
        rebuild_result = await _reindex_file_from_parsed(
            knowledge,
            file_id=bench_file_id,
            parsed_content_id=new_parsed_id,
        )
        payload = {
            "bench_pdf": str(pdf_path),
            "source_file_id": str(source_row.get("file_id") or ""),
            "source_filename": str(source_row.get("filename") or ""),
            "source_parsed_content_id": source_parsed_id,
            "bench_file_id": bench_file_id,
            "bench_filename": bench_filename,
            "bench_parsed_content_id": new_parsed_id,
            "reused_existing_bench_file": bool(reused_existing),
            **rebuild_result,
        }
        bench_results.append(payload)
        print(
            "[rebuild] bench file ready "
            f"bench_file_id={bench_file_id} reused={reused_existing} success={payload['success']} chunks={payload['num_chunks']}"
        )

    verify_file_ids = list(args.test_file_id) + [str(item["bench_file_id"]) for item in bench_results]
    pg_by_file = _verify_files_in_postgres(verify_file_ids)
    neo4j_by_file = _verify_chunks_in_neo4j(verify_file_ids)

    verification: list[dict[str, Any]] = []
    for file_id in verify_file_ids:
        pg = pg_by_file.get(file_id, {})
        neo = neo4j_by_file.get(file_id, {})
        combined = FileVerifyResult(
            file_id=file_id,
            owner_id=str(pg.get("owner_id") or ""),
            filename=str(pg.get("filename") or ""),
            status=str(pg.get("status") or "MISSING"),
            chunk_count=int(neo.get("chunk_count") or 0),
            chunk_with_page_start=int(neo.get("chunk_with_page_start") or 0),
            chunk_with_page_end=int(neo.get("chunk_with_page_end") or 0),
        )
        verification.append(asdict(combined))

    failed_index = [r for r in (test_results + bench_results) if not bool(r.get("success"))]
    bad_status = [v for v in verification if v.get("status") != "INDEXED"]
    bad_chunks = [
        v
        for v in verification
        if int(v.get("chunk_count") or 0) <= 0
        or int(v.get("chunk_with_page_start") or 0) != int(v.get("chunk_count") or 0)
        or int(v.get("chunk_with_page_end") or 0) != int(v.get("chunk_count") or 0)
    ]

    if failed_index or bad_status or bad_chunks:
        summary = {
            "failed_index": failed_index,
            "bad_status": bad_status,
            "bad_chunks": bad_chunks,
        }
        raise RuntimeError("Rebuild verification failed: " + json.dumps(summary, ensure_ascii=False))

    return RebuildResult(
        test_owner_id=str(test_owner_id),
        bench_owner_id=str(bench_owner_id),
        source_owner_id=str(source_owner_id),
        knowledge_config_path=str(os.getenv("KNOWLEDGE_CONFIG_PATH", "")),
        parser_parse_mode=str(os.getenv("PARSER_PARSE_MODE", "")),
        fast_rebuild_mode=bool(args.fast_rebuild_mode),
        reset_bench_owner=bool(args.reset_bench_owner),
        test_reindex_results=test_results,
        bench_rebuild_results=bench_results,
        verification=verification,
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild DeepSearch benchmark env for test + bench owners.")
    parser.add_argument("--test-owner-id", default=DEFAULT_TEST_OWNER_ID, help="Owner ID for existing test data.")
    parser.add_argument("--bench-owner-id", default=DEFAULT_BENCH_OWNER_ID, help="Owner ID for benchmark dataset.")
    parser.add_argument("--source-owner-id", default=DEFAULT_TEST_OWNER_ID, help="Owner ID used to copy parsed blobs.")
    parser.add_argument(
        "--test-file-id",
        action="append",
        dest="test_file_id",
        default=[],
        help="Test owner file_id to force reindex (repeatable).",
    )
    parser.add_argument(
        "--bench-pdf",
        action="append",
        dest="bench_pdf",
        default=[],
        help="Bench PDF path to ingest via parsed reuse (repeatable).",
    )
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Where to write the rebuild JSON report.",
    )
    parser.add_argument(
        "--reset-bench-owner",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Soft-delete existing bench owner files before rebuild.",
    )
    parser.add_argument(
        "--fast-rebuild-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a derived knowledge config that skips KG LLM extraction for stable rebuild speed.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.test_file_id:
        args.test_file_id = list(DEFAULT_TEST_FILE_IDS)
    if not args.bench_pdf:
        args.bench_pdf = list(DEFAULT_BENCH_PDFS)

    os.environ.setdefault("PARSER_PARSE_MODE", "native")

    try:
        result = asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001
        print(f"[rebuild] FAILED: {exc}")
        return 1

    report_path = Path(args.report_path)
    _ensure_parent(report_path)
    report_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rebuild] OK report={report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
