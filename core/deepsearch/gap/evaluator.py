"""Coverage evaluation utilities for DeepSearch."""
from dataclasses import dataclass
from typing import Iterable, List, Optional

from encapsulation.data_model.deepsearch import EvidenceChunk, GapDetectionResult


@dataclass
class GapDetectionSettings:
    """Thresholds that control whether external channels should trigger."""

    coverage_threshold: float = 0.7
    confidence_threshold: float = 0.6
    expected_min_chunks: int = 3


class GapDetectionEvaluator:
    """Compute coverage/confidence metrics for DeepSearch traces."""

    def __init__(self, settings: GapDetectionSettings | None = None):
        self.settings = settings or GapDetectionSettings()

    def evaluate(
        self,
        evidences: Iterable[EvidenceChunk],
        *,
        answer_confidence: Optional[float] = None,
        missing_topics: Optional[List[str]] = None,
    ) -> GapDetectionResult:
        evidence_list = list(evidences)
        coverage_score = self._coverage_score(evidence_list)
        # If the pipeline does not provide an explicit confidence estimate, avoid
        # triggering external search purely due to a missing value.
        confidence_score = answer_confidence if answer_confidence is not None else 1.0
        missing = missing_topics or []
        should_trigger = (
            coverage_score < self.settings.coverage_threshold
            or (answer_confidence is not None and confidence_score < self.settings.confidence_threshold)
        )
        reason = "coverage"
        if answer_confidence is not None and confidence_score < self.settings.confidence_threshold:
            reason = "confidence"
        if missing:
            reason = "missing_topics"
            should_trigger = True
        return GapDetectionResult(
            coverage_score=coverage_score,
            confidence_score=confidence_score,
            should_trigger_external=should_trigger,
            reason=reason,
            missing_topics=missing,
            diagnostics={
                "evidence_count": len(evidence_list),
                "expected_min_chunks": self.settings.expected_min_chunks,
            },
        )

    def _coverage_score(self, evidences: List[EvidenceChunk]) -> float:
        expected = max(self.settings.expected_min_chunks, 1)
        ratio = len(evidences) / expected
        return min(ratio, 1.0)
