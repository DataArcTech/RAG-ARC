"""Config for GapDetection evaluator."""
import os
from typing import Literal

from pydantic import Field

from core.deepsearch.gap import GapDetectionEvaluator, GapDetectionSettings
from framework.config import AbstractConfig


class GapDetectionEvaluatorConfig(AbstractConfig):
    """Builds GapDetectionEvaluator with threshold overrides."""

    type: Literal["gap_detection_evaluator"] = "gap_detection_evaluator"
    coverage_threshold: float = Field(0.7, description="Minimum acceptable coverage score")
    confidence_threshold: float = Field(0.6, description="Minimum acceptable answer confidence")
    expected_min_chunks: int = Field(3, description="Chunk count target for coverage normalization")

    def build(self) -> GapDetectionEvaluator:
        env = os.getenv
        coverage = float(env("DEEPSEARCH_GAP_COVERAGE_THRESHOLD") or self.coverage_threshold)
        confidence = float(env("DEEPSEARCH_GAP_CONFIDENCE_THRESHOLD") or self.confidence_threshold)
        min_chunks = int(env("DEEPSEARCH_GAP_EXPECTED_MIN_CHUNKS") or self.expected_min_chunks)

        settings = GapDetectionSettings(
            coverage_threshold=coverage,
            confidence_threshold=confidence,
            expected_min_chunks=min_chunks,
        )
        return GapDetectionEvaluator(settings=settings)
