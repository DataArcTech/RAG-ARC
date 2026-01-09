"""Cross-language file-scope query rewrite thresholds.

These heuristics decide whether to ask the LLM to rewrite a query into another language to better match
file names in scope. They must be configurable to avoid silent quality drift across data distributions.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FileScopeXlangThresholds:
    alpha_ratio_to_zh_min: float
    cjk_ratio_to_zh_max: float
    cjk_ratio_to_en_min: float
    alpha_ratio_to_en_max: float


def _coerce_ratio(raw: str | None, *, default: float) -> float:
    try:
        value = float(raw) if raw is not None else float(default)
    except (TypeError, ValueError):
        value = float(default)
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def load_file_scope_xlang_thresholds() -> FileScopeXlangThresholds:
    """Load thresholds from env vars with safe clamping."""

    return FileScopeXlangThresholds(
        alpha_ratio_to_zh_min=_coerce_ratio(
            os.getenv("DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_ZH_MIN"),
            default=0.25,
        ),
        cjk_ratio_to_zh_max=_coerce_ratio(
            os.getenv("DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_ZH_MAX"),
            default=0.05,
        ),
        cjk_ratio_to_en_min=_coerce_ratio(
            os.getenv("DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_EN_MIN"),
            default=0.15,
        ),
        alpha_ratio_to_en_max=_coerce_ratio(
            os.getenv("DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_EN_MAX"),
            default=0.08,
        ),
    )

