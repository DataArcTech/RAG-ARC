"""Gap Detection runtime computes evidence coverage and gates external channels."""
from typing import Any, Dict, Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk

from .evaluator import GapDetectionEvaluator


class GapDetectionEngine:
    """Evaluate reasoning traces and decide whether optional external channels should run."""

    def __init__(
        self,
        evaluator: GapDetectionEvaluator,
        *,
        telemetry_client: Any | None = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if evaluator is None:
            raise ValueError("GapDetectionEngine requires a GapDetectionEvaluator instance")
        self.evaluator = evaluator
        self.telemetry_client = telemetry_client
        self.config = config or {}

    def evaluate(self, reasoning_trace: Dict[str, Any]) -> Dict[str, Any]:
        """Return coverage metrics and the external trigger decision."""

        if not isinstance(reasoning_trace, dict):
            raise ValueError("reasoning_trace must be a dictionary")

        evidences = self._coerce_evidences(reasoning_trace.get("evidences") or [])
        coverage_metrics = reasoning_trace.get("coverage_metrics") or {}
        pending_external = reasoning_trace.get("pending_external") or []
        answer_confidence = self._extract_confidence(coverage_metrics, reasoning_trace)
        missing_topics = self._extract_missing_topics(coverage_metrics, reasoning_trace)
        think_trigger, think_missing = self._think_requests_external(reasoning_trace.get("think_notes") or [])
        if think_missing:
            merged = list({*missing_topics, *think_missing})
        else:
            merged = missing_topics

        result_model = self.evaluator.evaluate(
            evidences,
            answer_confidence=answer_confidence,
            missing_topics=merged,
        )
        payload = result_model.model_dump()
        external_allowed, external_decision = self._external_enabled()

        diagnostics = payload.setdefault("diagnostics", {})
        diagnostics.setdefault("evidence_count", len(evidences))
        diagnostics.setdefault("pending_external_steps", len(pending_external))
        diagnostics["external_allowed"] = external_allowed
        diagnostics["external_decision"] = external_decision
        diagnostics.setdefault("coverage_metrics", coverage_metrics)
        diagnostics.setdefault("think_gap_trigger", think_trigger)

        requested_trigger = bool(payload.get("should_trigger_external")) or think_trigger
        if requested_trigger and not external_allowed:
            payload["should_trigger_external"] = False
            payload["reason"] = "external_disabled"
            diagnostics["external_blocked"] = True
            if think_trigger:
                diagnostics["think_gap_blocked"] = True
        elif think_trigger and external_allowed:
            payload["should_trigger_external"] = True
            payload["reason"] = "think_gap"
            if think_missing:
                existing = set(payload.get("missing_topics") or [])
                existing.update(think_missing)
                payload["missing_topics"] = list(existing)

        self._log(payload, reasoning_trace)
        return payload

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_evidences(raw: Iterable[Any]) -> List[EvidenceChunk]:
        evidences: List[EvidenceChunk] = []
        for item in raw:
            if isinstance(item, EvidenceChunk):
                evidences.append(item)
                continue
            if not isinstance(item, dict):
                continue
            try:
                evidences.append(EvidenceChunk.model_validate(item))
            except Exception:
                continue
        return evidences

    @staticmethod
    def _extract_missing_topics(coverage_metrics: Dict[str, Any], reasoning_trace: Dict[str, Any]) -> List[str]:
        missing = coverage_metrics.get("missing_topics")
        extracted: List[str] = []
        if isinstance(missing, list):
            extracted.extend(str(topic) for topic in missing if str(topic).strip())
        think_notes = reasoning_trace.get("think_notes") or []
        for note in think_notes:
            topics = (
                (note.get("metadata") or {}).get("missing_topics")
                if isinstance(note, dict)
                else None
            )
            if isinstance(topics, list):
                extracted.extend(str(topic) for topic in topics if str(topic).strip())
        return extracted

    @staticmethod
    def _extract_confidence(coverage_metrics: Dict[str, Any], reasoning_trace: Dict[str, Any]) -> Optional[float]:
        for key in ("answer_confidence", "confidence", "avg_confidence"):
            value = coverage_metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        steps = reasoning_trace.get("reasoning_steps") or []
        for step in reversed(steps):
            diagnostics = step.get("diagnostics") if isinstance(step, dict) else None
            if not diagnostics:
                continue
            value = diagnostics.get("confidence")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    @staticmethod
    def _think_requests_external(think_notes: List[Any]) -> tuple[bool, List[str]]:
        triggered = False
        missing: List[str] = []
        seen: set[str] = set()
        for note in think_notes:
            metadata = None
            if hasattr(note, "metadata"):
                metadata = getattr(note, "metadata")
            elif isinstance(note, dict):
                metadata = note.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if metadata.get("gap_trigger") is True:
                triggered = True
            topics = metadata.get("missing_topics")
            if isinstance(topics, list):
                for item in topics:
                    token = str(item).strip()
                    if token and token not in seen:
                        seen.add(token)
                        missing.append(token)
        return triggered, missing

    def _external_enabled(self) -> tuple[bool, Dict[str, Any]]:
        """Resolve whether external search is allowed (config is the single source of truth)."""

        config_flag = self.config.get("enable_external_on_gap")
        channel_flag = self.config.get("external_channel_enabled")
        config_enabled = bool(config_flag) and bool(channel_flag)
        return config_enabled, {"source": "config", "config_value": config_enabled}

    def _log(self, payload: Dict[str, Any], reasoning_trace: Dict[str, Any]) -> None:
        if not self.telemetry_client:
            return
        log_method = getattr(self.telemetry_client, "log_gap_detection", None) or getattr(
            self.telemetry_client, "log_gap_result", None
        )
        if not callable(log_method):
            return
        run_id = None
        graph_context = reasoning_trace.get("graph_context")
        if isinstance(graph_context, dict):
            metadata = graph_context.get("metadata") or {}
            if isinstance(metadata, dict):
                run_id = metadata.get("run_id")
        try:
            log_method(result=payload, context={"question": reasoning_trace.get("question"), "run_id": run_id})
        except Exception:
            # Telemetry must never break the main pipeline
            return
