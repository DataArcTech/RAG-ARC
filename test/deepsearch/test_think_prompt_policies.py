from core.prompts.deepsearch.tools import THINK_TOOL_SYSTEM_PROMPT_EN


def test_think_prompt_mentions_scoped_global_search_and_negative_policy() -> None:
    prompt = THINK_TOOL_SYSTEM_PROMPT_EN
    assert "locate" in prompt
    assert "toc.tree" in prompt
    assert "search.file" not in prompt
    # Negative conclusions must be based on full section/page reads, not snippets.
    assert "you MUST read the full relevant pages via read.pages" in prompt
