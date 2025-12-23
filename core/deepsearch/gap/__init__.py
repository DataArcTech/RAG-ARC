"""Gap detection primitives and runtime orchestration."""

from .evaluator import GapDetectionEvaluator, GapDetectionSettings
from .runtime import GapDetectionEngine

__all__ = [
    "GapDetectionEvaluator",
    "GapDetectionSettings",
    "GapDetectionEngine",
]
