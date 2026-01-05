"""Centralized warnings configuration for runtime entrypoints and tests.

Library modules should avoid process-level side effects (e.g. calling
`warnings.filterwarnings(...)` at import time). Entrypoints (API/CLI) and the
test harness can call `configure_runtime_warnings()` early to keep logs clean.
"""
import warnings


def configure_runtime_warnings() -> None:
    """Apply repo-wide warning filters expected for normal runtime."""

    warnings.filterwarnings(
        "ignore",
        message=r"builtin type SwigPy.* has no __module__ attribute",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"builtin type swigvarlink has no __module__ attribute",
        category=DeprecationWarning,
    )
