import json
import logging
from typing import Any, Dict

from core.deepsearch.utils.computable_classifier_llm import aclassify_computable_question
from core.deepsearch.utils.computable_questions import coerce_str_list, detect_computable_question
from config.core.deepsearch.runtime_messages import (
    COMPUTABLE_HARD_GATE_MESSAGE,
    GENERATION_MODE_MISSING_DETERMINISTIC_TOOLS,
)

logger = logging.getLogger(__name__)


class DeepSearchServiceRoutingMixin:
    def _resolve_deterministic_routing_config(self) -> Dict[str, Any]:
        if not isinstance(self.config, dict):
            return {}
        cfg = self.config.get("deterministic_routing") or {}
        return cfg if isinstance(cfg, dict) else {}

    async def _classify_question(self, *, question: str, routing_cfg: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(routing_cfg.get("classifier") or "llm").strip().lower()
        fail_on_error = bool(routing_cfg.get("fail_on_classifier_error"))
        llm = getattr(getattr(self, "planner", None), "llm_connector", None)
        llm_model = str(routing_cfg.get("llm_model_name") or "").strip() or None
        if llm_model is None and llm is not None:
            llm_cfg = getattr(llm, "config", None)
            candidate = getattr(llm_cfg, "low_cost_model_name", None) if llm_cfg is not None else None
            llm_model = str(candidate).strip() if isinstance(candidate, str) and candidate.strip() else None
        llm_temperature = float(routing_cfg.get("llm_temperature") or 0.0)

        if mode in {"llm", "hybrid"}:
            try:
                result = await aclassify_computable_question(
                    llm,
                    question=question,
                    model=llm_model,
                    temperature=llm_temperature,
                )
                return {
                    "source": "llm",
                    "model": llm_model,
                    "is_computable": bool(result.is_computable),
                    "reasons": list(result.reasons),
                    "suggested_tools": list(result.suggested_tools),
                }
            except Exception as exc:  # noqa: BLE001
                if mode == "llm" and fail_on_error:
                    raise
                llm_error = str(exc) or exc.__class__.__name__
                if mode == "llm":
                    return {
                        "source": "llm",
                        "model": llm_model,
                        "is_computable": False,
                        "reasons": ["classifier_error"],
                        "suggested_tools": [],
                        "error": llm_error,
                    }
                routing_cfg = dict(routing_cfg)
                routing_cfg.setdefault("_llm_error", llm_error)

        if mode not in {"heuristic", "hybrid"}:
            return {"source": mode, "is_computable": False, "reasons": ["unsupported_classifier_mode"], "suggested_tools": []}

        signals = detect_computable_question(
            question,
            keywords=coerce_str_list(routing_cfg.get("computable_keywords")),
            operators=coerce_str_list(routing_cfg.get("computable_operators")),
            policy=routing_cfg.get("computable_policy") if isinstance(routing_cfg.get("computable_policy"), dict) else None,
        )
        payload: Dict[str, Any] = {"source": "heuristic", **signals.to_dict(), "suggested_tools": []}
        if mode == "hybrid" and isinstance(routing_cfg.get("_llm_error"), str):
            payload["llm_error"] = routing_cfg["_llm_error"]
        return payload

    @staticmethod
    def _count_deterministic_tool_runs(reasoning_trace: Dict[str, Any]) -> int:
        tool_results = reasoning_trace.get("tool_results") or []
        if not isinstance(tool_results, list):
            return 0
        count = 0
        for entry in tool_results:
            if not isinstance(entry, dict):
                continue
            result = entry.get("result")
            if not isinstance(result, dict):
                continue
            if str(result.get("determinism") or "").strip().lower() == "deterministic":
                count += 1
        return count

    def _maybe_hard_gate_computable_question(
        self,
        *,
        question: str,
        reasoning_trace: Dict[str, Any],
        classification: Any,
        routing_cfg: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        if not (isinstance(routing_cfg, dict) and routing_cfg.get("enabled")):
            return None
        if not routing_cfg.get("hard_gate_missing_deterministic_tools"):
            return None
        if not isinstance(classification, dict) or not bool(classification.get("is_computable")):
            return None

        deterministic_runs = self._count_deterministic_tool_runs(reasoning_trace)
        if deterministic_runs > 0:
            return None

        msg = COMPUTABLE_HARD_GATE_MESSAGE
        structured_report = {
            "format_version": str(getattr(getattr(self, "reporter", None), "STRUCTURED_REPORT_VERSION", "2.0")),
            "title": question,
            "short_answer": msg,
            "summary": msg,
            "sections": [],
            "limitations": [
                "Computable-question hard gate: no deterministic tool evidence was produced for this run.",
            ],
            "next_steps": [
                "If you expect a numeric/time answer, ensure the underlying facts carry temporal/value properties and rerun.",
                "Rewrite the question to include explicit entities and predicates (e.g. policy name + 'effective date').",
            ],
            "citations": [],
            "evidence_index": [],
            "graph_evidence": {},
            "text": msg,
            "context": {},
            "generation": {"mode": GENERATION_MODE_MISSING_DETERMINISTIC_TOOLS},
        }
        return {
            "question": question,
            "answer": msg,
            "highlights": [],
            "evidences": [],
            "structured_report": structured_report,
            "metadata": {
                "deterministic_routing": {
                    "hard_gated": True,
                    "deterministic_tool_run_count": deterministic_runs,
                    "classification": dict(classification),
                }
            },
        }

