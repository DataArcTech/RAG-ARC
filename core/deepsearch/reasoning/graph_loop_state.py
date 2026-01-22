import asyncio
import contextvars
from typing import List

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.deepsearch.memory.plan_state import PlanState


_RUN_EVIDENCES: contextvars.ContextVar[List[EvidenceChunk] | None] = contextvars.ContextVar(
    "deepsearch_run_evidences",
    default=None,
)
_RUN_EVIDENCE_LOCK: contextvars.ContextVar[asyncio.Lock | None] = contextvars.ContextVar(
    "deepsearch_run_evidence_lock",
    default=None,
)
_RUN_TOTAL_STEPS: contextvars.ContextVar[int] = contextvars.ContextVar("deepsearch_run_total_steps", default=0)
_RUN_THINK_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("deepsearch_run_think_count", default=0)
_RUN_REFLECT_COUNT: contextvars.ContextVar[int] = contextvars.ContextVar("deepsearch_run_reflect_count", default=0)
_RUN_THINK_TOOL_SIGNATURES: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar(
    "deepsearch_run_think_tool_signatures",
    default=None,
)
_RUN_PLAN_STATE: contextvars.ContextVar[PlanState | None] = contextvars.ContextVar(
    "deepsearch_run_plan_state",
    default=None,
)


def _run_evidence_state() -> tuple[List[EvidenceChunk], asyncio.Lock]:
    evidences = _RUN_EVIDENCES.get()
    lock = _RUN_EVIDENCE_LOCK.get()
    if evidences is None or lock is None:
        raise RuntimeError("DeepSearch evidence state is not initialized")
    return evidences, lock


def _run_plan_state() -> PlanState:
    plan_state = _RUN_PLAN_STATE.get()
    if plan_state is None:
        raise RuntimeError("DeepSearch plan state is not initialized")
    return plan_state


__all__ = [
    "_RUN_EVIDENCES",
    "_RUN_EVIDENCE_LOCK",
    "_RUN_REFLECT_COUNT",
    "_RUN_THINK_COUNT",
    "_RUN_THINK_TOOL_SIGNATURES",
    "_RUN_TOTAL_STEPS",
    "_RUN_PLAN_STATE",
    "_run_evidence_state",
    "_run_plan_state",
]
