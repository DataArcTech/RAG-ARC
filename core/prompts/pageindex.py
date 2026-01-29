"""Prompt templates for PageIndex section summaries.

Note: We intentionally do NOT generate doc-level descriptions/profiles at index time. Doc routing
is handled by DeepSearch's `search.file` via evidence-driven chunk retrieval.
"""

SECTION_SUMMARY_SYSTEM_PROMPT = (
    "You are a concise summarizer for retrieval. "
    "Summaries must stay faithful to the provided section content. "
    "Use the same language as the content and output plain text only."
)

SECTION_SUMMARY_USER_PROMPT = (
    "Summarize the following document section in 1-3 sentences.\n"
    "Section title: {title}\n"
    "Section path: {path}\n"
    "Content:\n"
    "{content}"
)

__all__ = [
    "SECTION_SUMMARY_SYSTEM_PROMPT",
    "SECTION_SUMMARY_USER_PROMPT",
]
