"""Prompt templates for PageIndex section summaries and doc descriptions."""

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

DOC_DESCRIPTION_SYSTEM_PROMPT = (
    "You write brief document descriptions for retrieval routing. "
    "Stay factual and concise, using the same language as the input."
)

DOC_DESCRIPTION_USER_PROMPT = (
    "Write a short description (2-4 sentences) of this document using the section summaries.\n"
    "Document title: {title}\n"
    "Section summaries:\n"
    "{summaries}"
)


__all__ = [
    "SECTION_SUMMARY_SYSTEM_PROMPT",
    "SECTION_SUMMARY_USER_PROMPT",
    "DOC_DESCRIPTION_SYSTEM_PROMPT",
    "DOC_DESCRIPTION_USER_PROMPT",
]
