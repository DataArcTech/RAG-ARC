#!/usr/bin/env python3
from __future__ import annotations

"""
Trigger (re)indexing for all files under an owner_id.

This is a convenience script for bulk reindexing in dev/ops workflows. It is safe by default:
- it triggers the normal indexing pipeline (no monkey patches)
- it can optionally wait for TaskRuns to complete (Celery mode polls Redis)
"""

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Iterable, Optional


def _repo_root() -> Path:
    # scripts/indexing/* -> repo root is two levels up
    return Path(__file__).resolve().parents[2]


def _strip_optional_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv(dotenv_path: Path, *, override: bool = False) -> None:
    """Lightweight dotenv loader (keeps this script dependency-free)."""
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


def _chunked(items: list[str], batch_size: int) -> Iterable[list[str]]:
    if batch_size <= 0:
        yield items
        return
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _query_owner_file_ids(owner_id: uuid.UUID, *, status: Optional[str] = None) -> list[str]:
    """Fetch file_ids by owner directly from Postgres so we can trigger re-indexing in bulk."""
    from sqlalchemy import text

    from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig

    status_token = (status or "").strip().upper()
    where_status = ""
    params = {"owner_id": owner_id}
    if status_token:
        where_status = " AND status = :status"
        params["status"] = status_token

    db = PostgreSQLConfig().build()
    with db.engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT file_id "
                "FROM file_metadata "
                "WHERE owner_id = :owner_id AND status != 'DELETED'"
                + where_status
                + " ORDER BY updated_at DESC"
            ),
            params,
        ).fetchall()
    return [str(row[0]) for row in rows]


def _latest_task_run_id(knowledge, file_id: str) -> str | None:  # noqa: ANN001
    try:
        token = knowledge.task_queue.get_latest_task_run_id_for_resource(
            task_type="index_file",
            resource_id=file_id,
        )
        return str(token) if token else None
    except Exception:
        return None


async def _wait_for_runs(
    knowledge,  # noqa: ANN001
    *,
    file_ids: list[str],
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> None:
    """Wait for the *latest* TaskRuns of the given file_ids to reach a terminal state."""
    from encapsulation.message_queue.redis_task_queue import TaskState

    pending_states = {TaskState.PENDING.value, TaskState.RUNNING.value}
    terminal_states = {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}

    start = asyncio.get_running_loop().time()
    last_print = 0.0
    while True:
        now = asyncio.get_running_loop().time()
        elapsed = now - start
        if timeout_seconds > 0 and elapsed >= timeout_seconds:
            print(f"[warn] wait timeout reached: {timeout_seconds:.0f}s")
            break

        done = 0
        succeeded = 0
        failed = 0
        missing = 0

        for file_id in file_ids:
            run_id = _latest_task_run_id(knowledge, file_id)
            if not run_id:
                missing += 1
                continue
            run = knowledge.task_queue.get_task_run(run_id) or {}
            state = str(run.get("state") or "")
            if state in terminal_states:
                done += 1
                if state == TaskState.SUCCESS.value:
                    succeeded += 1
                else:
                    failed += 1
            elif state in pending_states:
                continue

        if now - last_print >= max(1.0, poll_interval_seconds):
            last_print = now
            print(
                f"[wait] done={done}/{len(file_ids)} success={succeeded} failed={failed} missing={missing} "
                f"elapsed={elapsed:.0f}s"
            )

        if done >= len(file_ids):
            break

        await asyncio.sleep(max(0.5, float(poll_interval_seconds)))


async def _trigger(
    owner_id: uuid.UUID,
    file_ids: list[str],
    *,
    batch_size: int,
    wait: bool,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> None:
    from framework.register import Register

    from config.application.knowledge_config import KnowledgeConfig

    # Register only the Knowledge module (avoid initializing rag_inference / retrievers).
    repo_root = _repo_root()
    config_path = os.getenv("KNOWLEDGE_CONFIG_PATH", "").strip()
    if not config_path:
        profile = str(os.getenv("MODEL_PROFILE", "api") or "api").strip().lower()
        config_path = "config/json_configs/knowledge_local.json" if profile == "local" else "config/json_configs/knowledge.json"
    resolved = Path(config_path)
    if not resolved.is_absolute():
        resolved = (repo_root / resolved).resolve()

    reg = Register()
    if "knowledge" not in reg.registrations:
        reg.register(config_path=str(resolved), app_name="knowledge", config_type=KnowledgeConfig)
    knowledge = reg.get_object("knowledge")

    use_celery = bool(getattr(knowledge, "_use_celery", lambda: False)())
    print(f"[info] TASK_QUEUE_MODE={os.getenv('TASK_QUEUE_MODE')} (knowledge.use_celery={use_celery})")

    total = len(file_ids)
    print(f"[info] triggering re-index for owner_id={owner_id} files={total} batch_size={batch_size}")

    for idx, batch in enumerate(_chunked(file_ids, batch_size), start=1):
        before = {fid: _latest_task_run_id(knowledge, fid) for fid in batch}
        print(f\"\\n[batch {idx}] trigger_indexing files={len(batch)} (force=True, wait=False)\")
        msg = await knowledge.trigger_indexing(batch, owner_id, force=True, wait=False)
        print(msg)

        if wait:
            after = {fid: _latest_task_run_id(knowledge, fid) for fid in batch}
            scheduled = [fid for fid in batch if after.get(fid) and after.get(fid) != before.get(fid)]
            if not scheduled:
                print(\"[warn] no new TaskRuns scheduled in this batch; skipping wait.\")
            else:
                await _wait_for_runs(
                    knowledge,
                    file_ids=scheduled,
                    poll_interval_seconds=float(poll_interval_seconds),
                    timeout_seconds=float(timeout_seconds),
                )

        await asyncio.sleep(0.2)


def main(argv: list[str]) -> int:
    repo_root = _repo_root()
    default_env = repo_root / ".env"

    ap = argparse.ArgumentParser(description="Trigger re-indexing for all files under an owner_id.")
    ap.add_argument("--owner-id", required=True, type=_parse_uuid)
    ap.add_argument("--dotenv", default=str(default_env))
    ap.add_argument("--status", default=None, help="Optional file status filter (e.g., INDEXED).")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--wait", action="store_true", help="Poll TaskRuns until each batch completes.")
    ap.add_argument("--poll-interval-seconds", type=float, default=5.0)
    ap.add_argument("--timeout-seconds", type=float, default=0.0, help="0 means no timeout.")
    args = ap.parse_args(argv)

    _load_dotenv(Path(args.dotenv), override=False)

    owner_id = args.owner_id
    file_ids = _query_owner_file_ids(owner_id, status=args.status)
    if not file_ids:
        print("[info] no files found for owner; nothing to do.")
        return 0

    asyncio.run(
        _trigger(
            owner_id,
            file_ids,
            batch_size=int(args.batch_size),
            wait=bool(args.wait),
            poll_interval_seconds=float(args.poll_interval_seconds),
            timeout_seconds=float(args.timeout_seconds),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

