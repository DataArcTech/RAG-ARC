"""Atomic (non-splittable) units for file chunking.

These helpers are shared by chunkers to ensure certain Markdown constructs are
never split across chunk boundaries (e.g., fenced code blocks, tables).
"""

from .fenced_code import (  # noqa: F401
    SegmentKind,
    is_fence_line,
    parse_fenced_code_block,
    split_fenced_code_blocks,
    split_fenced_code_blocks_with_line_spans,
)
from .markdown_table import (  # noqa: F401
    TABLE_SEPARATOR_RE,
    extract_markdown_table_rows,
    is_markdown_table_start,
    parse_markdown_table,
)

