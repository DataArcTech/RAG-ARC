"""
MQ tool: sync RedisTaskQueue streams to PostgreSQL (daemon or one-shot).
"""

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.encapsulation.database.cache_db.redis_config import RedisConfig
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from encapsulation.data_model.orm_models import (
    TaskProgressEvent,
    TaskRun,
    TaskRunState,
    TaskSyncOffset,
)
from encapsulation.database.cache_db.redis_db import RedisDB
from encapsulation.message_queue.redis_task_queue import RedisTaskQueueSettings
from encapsulation.database.relational_db.postgresql import PostgreSQLDB

logger = logging.getLogger(__name__)


def _dt_from_ms(ts_ms: Optional[int]) -> Optional[datetime]:
    if ts_ms is None:
        return None
    try:
        return datetime.utcfromtimestamp(int(ts_ms) / 1000)
    except Exception:
        return None


def _parse_uuid_hex(value: Optional[str]) -> Optional[uuid.UUID]:
    if not value:
        return None
    value = str(value)
    try:
        if len(value) == 32:
            return uuid.UUID(hex=value)
        return uuid.UUID(value)
    except Exception:
        return None


def _load_offset(pg: PostgreSQLDB, stream: str) -> str:
    with pg.SessionMaker() as session:
        row = session.get(TaskSyncOffset, stream)
        return row.last_id if row else "0-0"


def _save_offset(pg: PostgreSQLDB, stream: str, last_id: str) -> None:
    now = datetime.utcnow()
    stmt = pg_insert(TaskSyncOffset.__table__).values(
        stream=stream,
        last_id=last_id,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[TaskSyncOffset.__table__.c.stream],
        set_={"last_id": last_id, "updated_at": now},
    )
    with pg.SessionMaker() as session:
        session.execute(stmt)
        session.commit()


def _chunks(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _upsert_task_runs(pg: PostgreSQLDB, records: list[Dict[str, Any]]) -> None:
    if not records:
        return

    # Redis Streams may contain multiple updates for the same task_run_id in a single XREAD batch.
    # PostgreSQL INSERT .. ON CONFLICT DO UPDATE cannot update the same target row more than once
    # within a single statement, so we de-duplicate by task_run_id (keeping the latest payload).
    deduped: dict[str, Dict[str, Any]] = {}
    for record in records:
        task_run_id = str(record.get("task_run_id") or "")
        if not task_run_id:
            continue
        deduped[task_run_id] = record
    records = list(deduped.values())
    if not records:
        return

    rows: list[Dict[str, Any]] = []
    for record in records:
        task_run_uuid = _parse_uuid_hex(str(record.get("task_run_id") or "")) or uuid.uuid4()
        owner_uuid = _parse_uuid_hex(record.get("owner_id"))

        state_raw = str(record.get("state") or TaskRunState.PENDING.value).upper()
        try:
            state = TaskRunState(state_raw)
        except Exception:
            state = TaskRunState.PENDING

        rows.append(
            {
                "task_run_id": task_run_uuid,
                "task_type": str(record.get("task_type") or "unknown"),
                "owner_id": owner_uuid,
                "resource_id": record.get("resource_id"),
                "state": state,
                "progress_percent": record.get("progress_percent"),
                "created_at": _dt_from_ms(record.get("created_at_ms")) or datetime.utcnow(),
                "updated_at": _dt_from_ms(record.get("updated_at_ms")) or datetime.utcnow(),
                "finished_at": _dt_from_ms(record.get("finished_at_ms")),
                "error_message": record.get("error_message"),
                "result_ref": record.get("result_ref"),
                "task_metadata": record.get("metadata") if isinstance(record.get("metadata"), dict) else {},
            }
        )

    stmt = pg_insert(TaskRun.__table__).values(rows)
    excluded = stmt.excluded
    stmt = stmt.on_conflict_do_update(
        index_elements=[TaskRun.__table__.c.task_run_id],
        set_={
            "task_type": excluded.task_type,
            "owner_id": excluded.owner_id,
            "resource_id": excluded.resource_id,
            "state": excluded.state,
            "progress_percent": excluded.progress_percent,
            "created_at": excluded.created_at,
            "updated_at": excluded.updated_at,
            "finished_at": excluded.finished_at,
            "error_message": excluded.error_message,
            "result_ref": excluded.result_ref,
            "task_metadata": excluded.task_metadata,
        },
    )

    with pg.SessionMaker() as session:
        session.execute(stmt)
        session.commit()


def _insert_progress_events(pg: PostgreSQLDB, events: list[Tuple[str, Dict[str, Any]]]) -> None:
    if not events:
        return

    # Ensure TaskRun rows exist (FK safety) without overwriting real data.
    placeholders: dict[uuid.UUID, Dict[str, Any]] = {}
    rows: list[Dict[str, Any]] = []
    for stream_id, event in events:
        task_run_uuid = _parse_uuid_hex(str(event.get("run_id") or "")) or uuid.uuid4()
        if task_run_uuid not in placeholders:
            placeholders[task_run_uuid] = {
                "task_run_id": task_run_uuid,
                "task_type": str(event.get("flow") or "unknown"),
                "owner_id": None,
                "resource_id": event.get("resource_id"),
                "state": TaskRunState.RUNNING,
                "progress_percent": None,
                "created_at": _dt_from_ms(event.get("ts_ms")) or datetime.utcnow(),
                "updated_at": _dt_from_ms(event.get("ts_ms")) or datetime.utcnow(),
                "finished_at": None,
                "error_message": None,
                "result_ref": None,
                "task_metadata": {},
            }
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        rows.append(
            {
                "stream_id": stream_id,
                "task_run_id": task_run_uuid,
                "flow": str(event.get("flow") or "unknown"),
                "resource_id": event.get("resource_id"),
                "seq": event.get("seq"),
                "ts_ms": event.get("ts_ms"),
                "stage": event.get("stage"),
                "status": event.get("status"),
                "percent": event.get("percent"),
                "payload": payload,
                "created_at": _dt_from_ms(event.get("ts_ms")) or datetime.utcnow(),
            }
        )

    stmt = pg_insert(TaskProgressEvent.__table__).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=[TaskProgressEvent.__table__.c.stream_id])

    with pg.SessionMaker() as session:
        if placeholders:
            ensure_stmt = pg_insert(TaskRun.__table__).values(list(placeholders.values()))
            ensure_stmt = ensure_stmt.on_conflict_do_nothing(index_elements=[TaskRun.__table__.c.task_run_id])
            session.execute(ensure_stmt)
        session.execute(stmt)
        session.commit()


def _read_stream_batch(
    client,
    *,
    stream: str,
    last_id: str,
    count: int,
    block_ms: int,
) -> tuple[str, list[tuple[str, dict[str, str]]]]:
    # Redis semantics: BLOCK 0 means block forever. For a non-blocking poll, omit BLOCK entirely.
    if block_ms and block_ms > 0:
        res = client.xread({stream: last_id}, count=count, block=block_ms)
    else:
        res = client.xread({stream: last_id}, count=count)
    if not res:
        return last_id, []
    # res: [(b'stream', [(b'id', {b'k': b'v'})])]; decode_responses=True -> str
    entries = res[0][1]
    if not entries:
        return last_id, []
    new_last = entries[-1][0]
    return new_last, entries


def sync_once(
    *,
    redis_db: RedisDB,
    pg: PostgreSQLDB,
    settings: RedisTaskQueueSettings,
    batch_size: int,
    block_ms: int,
) -> dict[str, int]:
    client = redis_db.client

    task_stream = settings.stream_task_runs()
    progress_stream = settings.stream_progress()

    task_last = _load_offset(pg, task_stream)
    prog_last = _load_offset(pg, progress_stream)

    synced_task_runs = 0
    synced_progress_events = 0

    new_task_last, task_entries = _read_stream_batch(
        client, stream=task_stream, last_id=task_last, count=batch_size, block_ms=block_ms
    )
    if task_entries:
        # De-duplicate within this pass (XREAD can return multiple updates for the same task_run_id).
        records_by_id: dict[str, Dict[str, Any]] = {}
        for _, fields in task_entries:
            payload = fields.get("payload")
            if not payload:
                continue
            try:
                record = json.loads(payload)
                if not isinstance(record, dict):
                    continue
                task_run_id = str(record.get("task_run_id") or "")
                if not task_run_id:
                    continue
                records_by_id[task_run_id] = record
            except Exception:
                continue
        for batch in _chunks(list(records_by_id.values()), 500):
            _upsert_task_runs(pg, batch)
            synced_task_runs += len(batch)
        _save_offset(pg, task_stream, new_task_last)

    new_prog_last, prog_entries = _read_stream_batch(
        client, stream=progress_stream, last_id=prog_last, count=batch_size, block_ms=block_ms
    )
    if prog_entries:
        parsed: list[Tuple[str, Dict[str, Any]]] = []
        for stream_id, fields in prog_entries:
            payload = fields.get("payload")
            if not payload:
                continue
            try:
                event = json.loads(payload)
                if isinstance(event, dict):
                    parsed.append((stream_id, event))
            except Exception:
                continue
        for batch in _chunks(parsed, 1000):
            _insert_progress_events(pg, batch)
            synced_progress_events += len(batch)
        _save_offset(pg, progress_stream, new_prog_last)

    return {"task_runs": synced_task_runs, "progress_events": synced_progress_events}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync Redis task queue records into PostgreSQL.")
    parser.add_argument("--once", action="store_true", help="Run one sync pass and exit.")
    parser.add_argument("--daemon", action="store_true", help="Run continuously (default).")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds to sleep between polls.")
    parser.add_argument("--batch-size", type=int, default=2000, help="Max Redis stream entries per pass, per stream.")
    parser.add_argument("--block-ms", type=int, default=1000, help="XREAD block time in ms for each pass.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = RedisTaskQueueSettings(
        namespace=os.getenv("MQ_NAMESPACE", "rag-arc:mq"),
        task_run_ttl_seconds=int(os.getenv("MQ_TASK_RUN_TTL_SECONDS", str(24 * 3600))),
        progress_ttl_seconds=int(os.getenv("MQ_PROGRESS_TTL_SECONDS", str(24 * 3600))),
        stream_maxlen=int(os.getenv("MQ_STREAM_MAXLEN", "20000")),
    )

    redis_db = RedisDB(RedisConfig())
    pg = PostgreSQLConfig().build()

    run_once = args.once and not args.daemon
    if run_once:
        while True:
            stats = sync_once(
                redis_db=redis_db,
                pg=pg,
                settings=settings,
                batch_size=max(1, args.batch_size),
                block_ms=0,
            )
            if stats["task_runs"] or stats["progress_events"]:
                logger.info("Synced: task_runs=%s progress_events=%s", stats["task_runs"], stats["progress_events"])
                continue
            return 0

    while True:
        stats = sync_once(
            redis_db=redis_db,
            pg=pg,
            settings=settings,
            batch_size=max(1, args.batch_size),
            block_ms=max(0, args.block_ms),
        )
        if stats["task_runs"] or stats["progress_events"]:
            logger.info("Synced: task_runs=%s progress_events=%s", stats["task_runs"], stats["progress_events"])
        time.sleep(max(0.1, args.poll_interval))


if __name__ == "__main__":
    raise SystemExit(main())
