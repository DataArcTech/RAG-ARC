from core.deepsearch.report.llm_writer import DeepSearchLLMReportWriter


def test_report_writer_prompt_style_deepsearch_includes_scope_policy() -> None:
    prompt = DeepSearchLLMReportWriter._system_prompt_with_style(
        "BASE",
        question="What is X?",
        context={"report_style": "deepsearch"},
    )
    assert "DeepSearch Scope Policy" in prompt


def test_report_writer_prompt_style_research_includes_research_hint_only() -> None:
    prompt = DeepSearchLLMReportWriter._system_prompt_with_style(
        "BASE",
        question="What is X?",
        context={"report_style": "research"},
    )
    assert "Research Report Style" in prompt
    assert "DeepSearch Scope Policy" not in prompt
