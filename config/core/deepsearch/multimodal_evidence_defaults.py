"""Defaults for DeepSearch multimodal evidence behavior.

Centralize constants/thresholds so multimodal behavior stays configurable and reviewable.
"""
# Heuristic cue keywords that suggest the user expects visual evidence (images/figures/tables).
# Keep this list short and domain-agnostic. It is used only to decide whether to *try* to
# preserve image chunks inside limited top-k evidence windows.
DEEPSEARCH_VISUAL_CUE_KEYWORDS: tuple[str, ...] = (
    # English
    "image",
    "images",
    "figure",
    "fig",
    "table",
    "chart",
    "diagram",
    "screenshot",
    "equation",
    # Chinese
    "图",
    "图片",
    "图表",
    "插图",
    "截图",
    "表",
    "表格",
    "公式",
)

# When visual cues are present and candidate image chunks exist, preserve at least this many
# images inside the requested top-k window by trading out the weakest non-image chunks.
DEEPSEARCH_VISUAL_MIN_IMAGES: int = 1

# Token extraction for Neo4j fallback search (when graph retrieval doesn't surface images).
DEEPSEARCH_VISUAL_QUERY_TOKEN_MIN_LEN: int = 2
DEEPSEARCH_VISUAL_QUERY_MAX_TOKENS: int = 8
DEEPSEARCH_VISUAL_FALLBACK_MAX_IMAGE_CANDIDATES: int = 12

# Over-fetch factor for retrieval when file scope is enabled or visual cues are present.
# Needed because image chunks often have weaker text similarity scores (caption-only index_text).
DEEPSEARCH_RETRIEVAL_OVERFETCH_FACTOR: int = 5

# Hard cap for over-fetching to keep Neo4j/embedding workloads bounded.
DEEPSEARCH_RETRIEVAL_OVERFETCH_MAX: int = 100
