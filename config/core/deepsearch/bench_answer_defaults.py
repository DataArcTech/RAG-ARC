"""Defaults for DeepSearch benchmark-mode answer synthesis.

Benchmark mode is used by offline evaluations (e.g., GraphRAG-Benchmark). It should be:
- configurable (no scattered constants / env-only knobs)
- reproducible (settings come from json config)
- conservative about evidence governance (prefer primary/citeable evidence by default)
"""
from typing import Dict, Literal, Tuple

BenchAnswerPreference = Literal["correctness", "coverage", "balanced"]
BenchAnswerMode = Literal["single_stage", "two_stage"]

DEFAULT_BENCH_ALLOWED_EVIDENCE_KINDS: Tuple[str, ...] = ("primary",)

# These defaults are intentionally modest; use `config/json_configs/deepsearch_service.json` to override per deployment.
DEFAULT_BENCH_MAX_EVIDENCE_ITEMS: int = 24
DEFAULT_BENCH_MAX_EVIDENCE_CHARS: int = 12000
DEFAULT_BENCH_SNIPPET_CHARS: int = 900

# Heuristic to preserve list/bullet details when the best match is a heading line.
DEFAULT_BENCH_HEADING_WINDOW_MAX_LINES: int = 10

# Question-type gating (GraphRAG-Benchmark uses these labels).
DEFAULT_BENCH_POLICIES_BY_QUESTION_TYPE: Dict[str, Dict[str, object]] = {
    "Complex Reasoning": {
        "mode": "two_stage",
        "preference": "correctness",
        "max_evidence_items": DEFAULT_BENCH_MAX_EVIDENCE_ITEMS,
        "max_evidence_chars": DEFAULT_BENCH_MAX_EVIDENCE_CHARS,
        "snippet_chars": DEFAULT_BENCH_SNIPPET_CHARS,
    },
    "Contextual Summarize": {
        "mode": "single_stage",
        "preference": "coverage",
        "max_evidence_items": DEFAULT_BENCH_MAX_EVIDENCE_ITEMS,
        "max_evidence_chars": DEFAULT_BENCH_MAX_EVIDENCE_CHARS,
        "snippet_chars": DEFAULT_BENCH_SNIPPET_CHARS,
    },
    "Fact Retrieval": {
        "mode": "single_stage",
        "preference": "correctness",
        "max_evidence_items": max(10, DEFAULT_BENCH_MAX_EVIDENCE_ITEMS // 2),
        "max_evidence_chars": max(4000, DEFAULT_BENCH_MAX_EVIDENCE_CHARS // 2),
        "snippet_chars": max(400, DEFAULT_BENCH_SNIPPET_CHARS // 2),
    },
    "Creative Generation": {
        "mode": "single_stage",
        "preference": "balanced",
        "max_evidence_items": DEFAULT_BENCH_MAX_EVIDENCE_ITEMS,
        "max_evidence_chars": DEFAULT_BENCH_MAX_EVIDENCE_CHARS,
        "snippet_chars": DEFAULT_BENCH_SNIPPET_CHARS,
    },
}

