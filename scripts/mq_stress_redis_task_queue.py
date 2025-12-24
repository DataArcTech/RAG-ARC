import argparse
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.encapsulation.database.cache_db.redis_config import RedisConfig
from encapsulation.database.cache_db.redis_db import RedisDB
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, RedisTaskQueueSettings


def _build_queue(namespace: str) -> RedisTaskQueue:
    settings = RedisTaskQueueSettings(
        namespace=namespace,
        task_run_ttl_seconds=3600,
        progress_ttl_seconds=3600,
        result_ttl_seconds=3600,
        stream_maxlen=50000,
    )
    return RedisTaskQueue(RedisConfig(), settings)


def _append_events(queue: RedisTaskQueue, run_id: str, *, n: int) -> None:
    for idx in range(1, n + 1):
        queue.append_progress_event(
            flow="stress",
            task_run_id=run_id,
            stage="progress",
            status="progress",
            percent=min(100, (idx * 100) // max(1, n)),
            resource_id=run_id,
            payload={"idx": idx},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="RedisTaskQueue stress (real Redis, small defaults).")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent run_ids.")
    parser.add_argument("--events-per-run", type=int, default=500, help="Progress events appended per run.")
    parser.add_argument("--workers", type=int, default=5, help="Thread pool size used for writers.")
    args = parser.parse_args()

    if args.runs <= 0 or args.events_per_run <= 0 or args.workers <= 0:
        raise SystemExit("runs/events-per-run/workers must be positive")

    namespace = f"rag-arc:mq:stress:{uuid.uuid4().hex}"
    queue = _build_queue(namespace)
    client = RedisDB(RedisConfig()).client

    run_ids: list[str] = []
    for _ in range(args.runs):
        run_id = uuid.uuid4().hex
        run_ids.append(run_id)
        queue.create_task_run(task_run_id=run_id, task_type="stress", owner_id=uuid.UUID(int=0), resource_id=run_id)

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_append_events, queue, run_id, n=args.events_per_run) for run_id in run_ids]
        for fut in as_completed(futures):
            fut.result()
    elapsed = time.time() - started

    total = args.runs * args.events_per_run
    print(f"wrote events: total={total} runs={args.runs} events_per_run={args.events_per_run} elapsed_s={elapsed:.3f}")

    for run_id in run_ids:
        latest = queue.get_latest_progress_event(run_id)
        if not latest or int(latest.get("seq") or 0) != args.events_per_run:
            raise SystemExit(f"latest seq mismatch (run_id={run_id}) latest={latest}")

    # Regression check: if seq_map is missing (expired/flushed), replay must still advance.
    for run_id in run_ids[: min(2, len(run_ids))]:
        seq_map_key = queue.settings.key_seq_map(run_id)
        client.delete(seq_map_key)
        last_seq = max(0, args.events_per_run - 10)
        events = queue.read_progress_events(run_id, last_seq=last_seq, count=10, block_ms=0)
        seqs = [int(ev.get("seq") or -1) for ev in events]
        expected = list(range(last_seq + 1, args.events_per_run + 1))
        if seqs != expected:
            raise SystemExit(f"seq_map-missing replay mismatch (run_id={run_id}) got={seqs} expected={expected}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

