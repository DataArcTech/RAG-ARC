#!/usr/bin/env bash
set -euo pipefail

# Concurrent intent-routing test (multi-user, shared router instance).
#
# - Uses local Qwen embeddings (offline; no downloads).
# - Spawns N concurrent "users" each running a multi-turn dialogue sequentially.
# - Validates intent/action per turn and checks for cross-session leakage.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export INTENT_QWEN_EMBEDDING_MODEL_NAME="${INTENT_QWEN_EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
export INTENT_EMBEDDING_CACHE_FOLDER="${INTENT_EMBEDDING_CACHE_FOLDER:-./models/Qwen}"

uv run python - <<'PY'
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from application.intent_routing import IntentRoutingService


def _history_text(lines: list[tuple[str, str]]) -> str | None:
    if not lines:
        return None
    return "\n".join(f"{role}: {content}" for role, content in lines if role and content) or None


@dataclass(frozen=True)
class Turn:
    user: str
    enable_web_search: bool
    expected_intent: str
    expected_action: str


def _scenario() -> list[Turn]:
    # Use the same "real user" multi-turn scenario as the full-pipeline e2e to validate:
    # - single-turn vs multi-turn follow-ups
    # - topic switch + return-to-topic behavior under concurrency
    return [
        Turn("为我分析下新加坡美国学校的优劣", False, "RAG_REQUIRED", "rag"),
        Turn("详细展开说说", False, "FOLLOWUP_NO_RAG", "no_retrieval"),
        Turn("你这个回答不对，太泛了", False, "ANSWER_DISSATISFIED", "no_retrieval"),
        Turn("今天天气咋样", True, "WEB_ONLY", "web_only"),
        Turn("那新加坡的天气呢", True, "WEB_ONLY", "web_only"),
        # Return to the original topic ("新加坡美国学校") after switching to weather.
        Turn("如果我想要上新加坡美国学校，需要什么准备", False, "RAG_REQUIRED", "rag"),
        Turn("谢谢", False, "NO_RETRIEVAL", "no_retrieval"),
    ]


def _run_one_user(shared: IntentRoutingService, user_no: int) -> dict:
    session_id = f"user{user_no}-{uuid.uuid4()}"
    history: list[tuple[str, str]] = []
    per_ms: list[float] = []
    failures: list[str] = []
    saw_switch = False
    saw_return = False

    for i, turn in enumerate(_scenario(), start=1):
        t0 = time.perf_counter()
        res = shared.route(
            session_id=session_id,
            user_query=turn.user,
            history_text=_history_text(history),
            enable_web_search=turn.enable_web_search,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        per_ms.append(dt_ms)
        got_intent = str(res.intent)
        got_action = str(res.action)
        topic_action = getattr(res.topic, "action", None) if res.topic is not None else None
        if got_intent == "ANSWER_DISSATISFIED" and topic_action != "same_topic":
            failures.append(
                f"user#{user_no} turn#{i} topic mismatch for dissatisfaction: expected=same_topic got={topic_action} user={turn.user!r}"
            )
        if topic_action == "topic_switch":
            saw_switch = True
        if topic_action == "return_to_topic":
            saw_return = True

        if got_intent != turn.expected_intent:
            failures.append(
                f"user#{user_no} turn#{i} intent mismatch: expected={turn.expected_intent} got={got_intent} user={turn.user!r}"
            )
        if got_action != turn.expected_action:
            failures.append(
                f"user#{user_no} turn#{i} action mismatch: expected={turn.expected_action} got={got_action} user={turn.user!r}"
            )

        # Keep both roles in history; router must ignore assistant for topic/intent.
        history.append(("user", turn.user))
        history.append(("assistant", "OK (concurrency placeholder reply)"))

    return {
        "user_no": user_no,
        "session_id": session_id,
        "ms": per_ms,
        "failures": failures,
        "saw_topic_switch": saw_switch,
        "saw_return_to_topic": saw_return,
    }


def main() -> int:
    # Use CUDA if available unless pinned by caller.
    if not os.environ.get("INTENT_EMBEDDING_DEVICE"):
        try:
            import torch

            if torch.cuda.is_available():
                os.environ["INTENT_EMBEDDING_DEVICE"] = "cuda"
        except Exception:
            pass

    n_users = int(os.environ.get("INTENT_CONCURRENCY_USERS", "24"))
    workers = int(os.environ.get("INTENT_CONCURRENCY_WORKERS", "8"))
    print(f"[cfg] users={n_users} workers={workers} device={os.environ.get('INTENT_EMBEDDING_DEVICE','')}")

    cfg_path = os.getenv("INTENT_ROUTER_CONFIG_PATH", "config/core/intent_routing/intent_router.toml")
    t_init = time.perf_counter()
    svc = IntentRoutingService(config_path=cfg_path)
    init_ms = (time.perf_counter() - t_init) * 1000.0
    print(f"[init] config={cfg_path} ms={init_ms:.1f}")

    all_ms: list[float] = []
    failures: list[str] = []
    missing_switch = 0
    missing_return = 0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_one_user, svc, i) for i in range(1, n_users + 1)]
        for fut in as_completed(futs):
            r = fut.result()
            all_ms.extend(r["ms"])
            failures.extend(r["failures"])
            if not r["saw_topic_switch"]:
                missing_switch += 1
            if not r["saw_return_to_topic"]:
                missing_return += 1

    wall_ms = (time.perf_counter() - t0) * 1000.0
    p50 = statistics.median(all_ms) if all_ms else 0.0
    p95 = sorted(all_ms)[max(int(len(all_ms) * 0.95) - 1, 0)] if all_ms else 0.0
    avg = statistics.mean(all_ms) if all_ms else 0.0
    qps = (len(all_ms) / (wall_ms / 1000.0)) if wall_ms > 0 else 0.0

    print(
        f"[metrics] calls={len(all_ms)} wall_ms={wall_ms:.1f} qps={qps:.2f} p50_ms={p50:.2f} p95_ms={p95:.2f} avg_ms={avg:.2f}"
    )
    print(f"[topic] users_missing_topic_switch={missing_switch}/{n_users} users_missing_return_to_topic={missing_return}/{n_users}")

    if failures:
        print("[fail] mismatches (first 20):")
        for msg in failures[:20]:
            print(" - " + msg)
        return 2

    print("[ok] concurrency intent routing passed")
    return 0


raise SystemExit(main())
PY
