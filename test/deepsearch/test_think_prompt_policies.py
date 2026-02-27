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


def test_think_prompt_has_evidence_sufficiency_check() -> None:
    prompt = THINK_TOOL_SYSTEM_PROMPT_EN
    assert "## Evidence Sufficiency Self-Check" in prompt
    assert "Can I already answer" in prompt


def test_think_prompt_has_read_pages_precision() -> None:
    prompt = THINK_TOOL_SYSTEM_PROMPT_EN
    assert "## read.pages" in prompt
    assert "1-3 pages" in prompt
    assert "never read >5 pages" in prompt


def test_think_prompt_has_unanswerable_fast_exit() -> None:
    prompt = THINK_TOOL_SYSTEM_PROMPT_EN
    assert "Unanswerable fast-exit" in prompt
    assert "toc.tree shows NO section relevant" in prompt
    assert "information not found" in prompt


def test_read_pages_soft_cap_default() -> None:
    from config.core.deepsearch import tool_defaults
    assert tool_defaults.READ_PAGES_SOFT_CAP_ADVISORY == 3


def test_think_prompt_has_plan_guidance_section() -> None:
    prompt = THINK_TOOL_SYSTEM_PROMPT_EN
    assert "## Plan Guidance" in prompt
    assert "current_plan" in prompt
    assert "checked=true" in prompt
    assert "NOT sequential gates" in prompt
    # Plan Guidance should NOT duplicate code.python section content
    plan_section_start = prompt.index("## Plan Guidance")
    plan_section_end = prompt.index("##", plan_section_start + 1)
    plan_section = prompt[plan_section_start:plan_section_end]
    assert "code.python" not in plan_section
