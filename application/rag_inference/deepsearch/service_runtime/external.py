import json
import logging
from typing import Any, Dict, Optional, Sequence

from core.deepsearch.trace import emit_trace

logger = logging.getLogger(__name__)


class DeepSearchServiceExternalMixin:
    def _evaluate_gap(self, reasoning_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        gap_detector = getattr(self, "gap_detector", None)
        if not gap_detector:
            return None
        result = gap_detector.evaluate(reasoning_trace)
        return result if isinstance(result, dict) else None

    def _gap_stage(self, *, reasoning_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self._evaluate_gap(reasoning_trace)

    def _external_policy(self) -> Dict[str, Any]:
        channel = getattr(self, "external_channel", None)
        cfg = getattr(channel, "config", None)
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _autogenerate_external_task_if_needed(
        self,
        *,
        question: str,
        gap_result: Optional[Dict[str, Any]],
        pending_tasks: Sequence[Dict[str, Any]],
        reason: str,
    ) -> list[Dict[str, Any]]:
        if pending_tasks:
            return list(pending_tasks)
        if not gap_result or not gap_result.get("should_trigger_external"):
            return []
        if not self._external_allowed_flag():
            return []

        policy = self._external_policy()
        if policy.get("auto_generate_task_on_gap") is False:
            return []

        tool_name = self._tool_names()["web_channel_tool"]
        missing_topics = gap_result.get("missing_topics") if isinstance(gap_result.get("missing_topics"), list) else []
        suffix = ""
        if missing_topics:
            suffix = " " + " ".join([str(t).strip() for t in missing_topics if str(t).strip()][:6])
        query = (str(question or "").strip() + suffix).strip()
        if not query:
            return []

        return [
            {
                "step_id": "auto_web_01",
                "description": f"Auto external search ({reason}).",
                "channel": "web",
                "tool": tool_name,
                "tool_args": {"query": query, "reason": reason},
                "metadata": {
                    "provider": getattr(self.external_channel, "default_provider", None) if getattr(self, "external_channel", None) else None,
                    "source": "gap_detection",
                    "auto_generated": True,
                    "reason": reason,
                },
                "requires_external": True,
                "enabled": True,
            }
        ]

    async def _run_external_if_needed(
        self,
        *,
        gap_result: Optional[Dict[str, Any]],
        reasoning_trace: Dict[str, Any],
    ) -> Optional[Dict[str, Sequence[Dict[str, Any]]]]:
        if not getattr(self, "external_channel", None):
            return None
        if not self._external_allowed_flag():
            return None

        raw_tasks = list(reasoning_trace.get("pending_external") or [])
        policy = self._external_policy()

        should_trigger_gap = bool(gap_result and gap_result.get("should_trigger_external"))
        forced_tasks: list[Dict[str, Any]] = []
        for task in raw_tasks:
            if not isinstance(task, dict):
                continue
            metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
            if metadata.get("force_external") is True or metadata.get("requires_external_reason") in {"realtime", "latest"}:
                forced_tasks.append(task)

        tasks: list[Dict[str, Any]] = []
        reason = None
        if should_trigger_gap:
            tasks = self._autogenerate_external_task_if_needed(
                question=str(reasoning_trace.get("question") or "").strip(),
                gap_result=gap_result,
                pending_tasks=raw_tasks,
                reason="gap_detected",
            )
            reason = "gap_detected"
        elif forced_tasks and policy.get("execute_forced_tasks_without_gap", True):
            tasks = forced_tasks
            reason = "forced_by_policy"
        elif raw_tasks and policy.get("execute_pending_tasks_without_gap") is True:
            tasks = [t for t in raw_tasks if isinstance(t, dict)]
            reason = "pending_without_gap"

        if not tasks:
            return None

        try:
            await emit_trace(
                "think",
                "\n".join(
                    [
                        "External channel triggered.",
                        f"reason={reason}",
                        f"task_count={len(tasks)}",
                        f"auto_generated={any(bool(t.get('metadata', {}).get('auto_generated')) for t in tasks if isinstance(t, dict))}",
                        f"forced_count={len(forced_tasks)}",
                        f"pending_count={len(raw_tasks)}",
                    ]
                ),
                meta={"stage": "external_channel", "reason": reason, "task_count": len(tasks), "gap_result": gap_result},
            )
        except Exception:
            pass

        return await self.external_channel.run(  # type: ignore[union-attr]
            tasks,
            reasoning_trace=reasoning_trace,
            gap_result=gap_result if should_trigger_gap else None,
        )

    async def _run_external_tasks_direct(
        self,
        *,
        tasks: Sequence[Dict[str, Any]],
        reasoning_trace: Dict[str, Any],
        gap_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Sequence[Dict[str, Any]]]]:
        if not tasks or not getattr(self, "external_channel", None):
            return None
        if not self._external_allowed_flag():
            return None
        return await self.external_channel.run(list(tasks), reasoning_trace=reasoning_trace, gap_result=gap_result)  # type: ignore[union-attr]
