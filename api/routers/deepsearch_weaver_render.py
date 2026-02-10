"""Back-compat re-export for DeepSearch weaver rendering utilities.

The implementation lives in `core.presentation.deepsearch_weaver_render` so it can be
used by both API streaming and offline scripts without importing API routers.
"""

from core.presentation.deepsearch_weaver_render import (  # noqa: F401
    format_weaver_progress,
    render_trace_payload,
    weaver_block,
)

__all__ = [
    "format_weaver_progress",
    "render_trace_payload",
    "weaver_block",
]

