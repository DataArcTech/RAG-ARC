from core.prompts.deepsearch.tools import THINK_TOOL_SYSTEM_PROMPT_EN


def test_think_prompt_mentions_tools_and_evidence_policy() -> None:
    prompt = THINK_TOOL_SYSTEM_PROMPT_EN
    assert "locate" in prompt
    assert "toc.tree" in prompt
    assert "read.pages" in prompt
    assert "code.python" in prompt
    assert "web.search" in prompt
    assert "report_needed=false" in prompt
    assert "search.file" not in prompt
    # Numeric questions should read full table pages.
    assert "read the full table pages via read.pages" in prompt
    # regex_patterns should have its own section
    assert "## regex_patterns" in prompt
    assert "regex_patterns" in prompt
    # is_final should have its own section
    assert "## Finishing" in prompt
    assert "is_final=true" in prompt
    # code.python should have its own section, discouraging mental math
    assert "## code.python" in prompt
    assert "Mental math" in prompt
    # Anti-pattern: re-reading same pages
    assert "re-reading the same pages" in prompt
