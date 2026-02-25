"""Prompt templates for PageIndex section summaries.

PageIndex focuses on long-document navigation (section tree + section summaries). DeepSearch handles
relevant-file routing online via `locate`.
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
