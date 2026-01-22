"""DeepSearch progress semantics shared across API and Celery.

Progress is intentionally coarse: stage transitions are discrete, while the `reasoned`
stage interpolates based on step_count and completed_steps emitted by the
graph reasoning loop when available.
"""
STAGE_ORDER: tuple[str, ...] = (
    "created",
    "planned",
    "reasoned",
    "reported",
    "done",
    "failed",
)

# Base stage → percent mapping. Keep stable to avoid front-end churn.
STAGE_PERCENT_BASE: dict[str, int] = {
    "created": 0,
    "planned": 10,
    # `reasoned` uses interpolation between planned and reported when counts are available.
    # Fallback stays at this value to avoid UI regressions.
    "reasoned": 40,
    "reported": 80,
    "done": 100,
    "failed": 100,
}

REASONED_START_PERCENT: int = STAGE_PERCENT_BASE["planned"]
REASONED_END_PERCENT: int = STAGE_PERCENT_BASE["reported"]
