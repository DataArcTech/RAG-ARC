#!/usr/bin/env bash
set -euo pipefail

# E2E intent-routing smoke test (local Qwen embeddings + real Redis).
#
# What this tests:
# - Multi-turn intent classification (semantic-router based).
# - Session-scoped topic stack behavior: topic_switch and return_to_topic.
# - Latency/throughput basics for intent routing only (no RAG pipeline).
#
# Requirements:
# - Redis reachable via the project's RedisConfig (default localhost:6379).
# - Local model cache already present (no downloads). We enforce offline mode.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Local intent embedding model (decoupled from RAG embeddings).
export INTENT_QWEN_EMBEDDING_MODEL_NAME="${INTENT_QWEN_EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"
export INTENT_EMBEDDING_CACHE_FOLDER="${INTENT_EMBEDDING_CACHE_FOLDER:-./models/Qwen}"

uv run python - <<'PY'
import os
import statistics
import time
import uuid
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


def main() -> int:
    # If caller didn't pin a device, use CUDA when available for realistic latency.
    if not os.environ.get("INTENT_EMBEDDING_DEVICE"):
        try:
            import torch

            if torch.cuda.is_available():
                os.environ["INTENT_EMBEDDING_DEVICE"] = "cuda"
        except Exception:
            pass

    model_name = os.environ.get("INTENT_QWEN_EMBEDDING_MODEL_NAME", "")
    cache_folder = os.environ.get("INTENT_EMBEDDING_CACHE_FOLDER", "")
    device = os.environ.get("INTENT_EMBEDDING_DEVICE", "")
    print(f"[env] INTENT_QWEN_EMBEDDING_MODEL_NAME={model_name}")
    print(f"[env] INTENT_EMBEDDING_CACHE_FOLDER={cache_folder}")
    print(f"[env] INTENT_EMBEDDING_DEVICE={device}")

    # One session: we expect topic_switch + return_to_topic within the same session id.
    session_id = str(uuid.uuid4())

    # Complex multi-turn conversation:
    # - Topic A: PDF/doc questions (RAG_REQUIRED + followups)
    # - Topic B: programming/web questions (topic switch)
    # - Return to topic A after many turns (return_to_topic) and exceed the last-10-user-message window.
    turns: list[Turn] = [
        Turn(
            user="这个是什么意思？",
            enable_web_search=False,
            expected_intent="CLARIFY_REQUIRED",
            expected_action="no_retrieval",
        ),
        Turn(
            user="把下面这段话改写得更清晰：我想要一个可以跑得更快的程序，但是我也不想让代码太复杂。",
            enable_web_search=False,
            expected_intent="TASK_EXECUTION",
            expected_action="no_retrieval",
        ),
        Turn(
            user="请根据我上传的资料回答：test/test_pdf.pdf 的主要内容是什么？",
            enable_web_search=False,
            expected_intent="RAG_REQUIRED",
            expected_action="rag",
        ),
        Turn(
            user="你刚刚说的是什么意思？请结合 test/test_pdf.pdf 的内容再解释一下（同一个问题）。",
            enable_web_search=False,
            expected_intent="FOLLOWUP_NO_RAG",
            expected_action="no_retrieval",
        ),
        Turn(
            user="继续上一个问题：关于 test/test_pdf.pdf，请再从资料里找一下证据（最好带条款/页码）。",
            enable_web_search=False,
            expected_intent="FOLLOWUP_RAG_REQUIRED",
            expected_action="rag",
        ),
        Turn(
            user="你这个回答不对，太泛了，重新回答",
            enable_web_search=False,
            expected_intent="ANSWER_DISSATISFIED",
            expected_action="no_retrieval",
        ),
        Turn(
            user="换个话题：用 Python 写一个快速排序（quick sort），给我代码。",
            enable_web_search=False,
            expected_intent="TASK_EXECUTION",
            expected_action="no_retrieval",
        ),
        Turn(
            user="继续刚才快速排序：再展开讲讲 quick sort 的时间复杂度和空间复杂度。",
            enable_web_search=False,
            expected_intent="FOLLOWUP_NO_RAG",
            expected_action="no_retrieval",
        ),
        Turn(
            user="帮我网上查一下：快速排序（quick sort）是谁提出的？给我来源链接。",
            enable_web_search=True,
            expected_intent="WEB_ONLY",
            expected_action="web_only",
        ),
        # exceed last-10-user-message window with more B-topic turns
        Turn(
            user="继续刚才快速排序：quick sort 和归并排序各自适用什么场景？",
            enable_web_search=False,
            expected_intent="FOLLOWUP_NO_RAG",
            expected_action="no_retrieval",
        ),
        Turn(
            user="继续快速排序：为什么 quick sort 在平均情况下更快？",
            enable_web_search=False,
            expected_intent="FOLLOWUP_NO_RAG",
            expected_action="no_retrieval",
        ),
        Turn(
            user="还是快速排序：再给我一个你自己的例子说明分治法。",
            enable_web_search=False,
            expected_intent="FOLLOWUP_NO_RAG",
            expected_action="no_retrieval",
        ),
        Turn(
            user="谢谢，关于 quick sort 我明白了。",
            enable_web_search=False,
            expected_intent="NO_RETRIEVAL",
            expected_action="no_retrieval",
        ),
        Turn(
            user="好的我知道了（关于 quick sort）。",
            enable_web_search=False,
            expected_intent="NO_RETRIEVAL",
            expected_action="no_retrieval",
        ),
        # Return to topic A (should be return_to_topic; intent should require RAG)
        Turn(
            user="回到刚才的主题（test/test_pdf.pdf），请再从资料里找一下证据。",
            enable_web_search=False,
            expected_intent="FOLLOWUP_RAG_REQUIRED",
            expected_action="rag",
        ),
    ]

    cfg_path = os.getenv("INTENT_ROUTER_CONFIG_PATH", "config/core/intent_routing/intent_router.toml")
    t0 = time.perf_counter()
    svc = IntentRoutingService(config_path=cfg_path)
    init_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[init] config={cfg_path} IntentRoutingService init_ms={init_ms:.1f}")

    history: list[tuple[str, str]] = []
    per_turn_ms: list[float] = []
    saw_topic_switch = False
    saw_return_to_topic = False
    failures: list[str] = []

    for i, turn in enumerate(turns, start=1):
        t1 = time.perf_counter()
        res = svc.route(
            session_id=session_id,
            user_query=turn.user,
            history_text=_history_text(history),
            enable_web_search=turn.enable_web_search,
        )
        dt_ms = (time.perf_counter() - t1) * 1000.0
        per_turn_ms.append(dt_ms)

        got_intent = str(res.intent)
        got_action = str(res.action)
        got_topic_action = None
        if res.topic is not None:
            got_topic_action = str(getattr(res.topic, "action", None) or "") or None
        if got_topic_action == "topic_switch":
            saw_topic_switch = True
        if got_topic_action == "return_to_topic":
            saw_return_to_topic = True

        ok = True
        if got_intent == "ANSWER_DISSATISFIED" and got_topic_action != "same_topic":
            ok = False
            failures.append(
                f"turn#{i} topic mismatch for dissatisfaction: expected=same_topic got={got_topic_action} user={turn.user!r}"
            )
        if got_intent != turn.expected_intent:
            ok = False
            failures.append(f"turn#{i} intent mismatch: expected={turn.expected_intent} got={got_intent} user={turn.user!r}")
        if got_action != turn.expected_action:
            ok = False
            failures.append(f"turn#{i} action mismatch: expected={turn.expected_action} got={got_action} user={turn.user!r}")
        print(
            {
                "turn": i,
                "intent": got_intent,
                "action": got_action,
                "topic_action": got_topic_action,
                "score": round(float(res.score or 0.0), 4),
                "ms": round(dt_ms, 2),
                "ok": ok,
            }
        )

        # Append both user and assistant lines; topic stack must ignore assistant turns for classification.
        history.append(("user", turn.user))
        history.append(("assistant", "OK（e2e placeholder reply, should be ignored by intent/topic classification）"))

    # Metrics
    intent_acc = 1.0 if not failures else 0.0
    p50 = statistics.median(per_turn_ms) if per_turn_ms else 0.0
    p95 = sorted(per_turn_ms)[max(int(len(per_turn_ms) * 0.95) - 1, 0)] if per_turn_ms else 0.0
    avg = statistics.mean(per_turn_ms) if per_turn_ms else 0.0
    print(f"[metrics] turns={len(turns)} intent_action_topic_all_correct={intent_acc:.0f} p50_ms={p50:.2f} p95_ms={p95:.2f} avg_ms={avg:.2f}")

    if not saw_topic_switch:
        failures.append("topic stack never produced topic_switch in this scenario (expected at least once)")
    if not saw_return_to_topic:
        failures.append("topic stack never produced return_to_topic in this scenario (expected at least once)")

    if failures:
        print("[fail] mismatches:")
        for msg in failures:
            print(" - " + msg)
        return 2

    print("[ok] all checks passed")
    return 0


raise SystemExit(main())
PY
