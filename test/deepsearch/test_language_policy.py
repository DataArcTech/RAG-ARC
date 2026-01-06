import pytest


def test_infer_user_language_ignores_file_lists_and_titles():
    from core.deepsearch.utils.language_policy import infer_user_language

    q = (
        "What is the premium rebate schedule and the prepayment interest rate?\n"
        "Files:\n"
        "- docs-proj/英文港险产品/AXA安盛「盛利II-至尊」保费回赠及预缴利率 截止至12月31日（英文版）.pdf\n"
    )
    assert infer_user_language(q) == "en"

    q2 = "请根据《AXA安盛「盛利II-至尊」》说明一下保费回赠与预缴利率。"
    assert infer_user_language(q2) == "zh"


def test_report_writer_language_enforcement_uses_inferred_user_language():
    from core.deepsearch.report.llm_writer import DeepSearchLLMReportWriter

    base = "BASE_PROMPT"
    q = (
        "What is the premium rebate schedule?\n"
        "Files:\n"
        "- AXA安盛「盛利II-至尊」.pdf\n"
    )
    sys_prompt = DeepSearchLLMReportWriter._system_prompt_with_language(base, question=q)
    assert "Write ALL fields in English" in sys_prompt
    assert "Do NOT output" in sys_prompt

    q2 = "请总结这份英文产品文件的核心信息。"
    sys_prompt2 = DeepSearchLLMReportWriter._system_prompt_with_language(base, question=q2)
    assert "Write ALL fields in Simplified Chinese" in sys_prompt2


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What is the schedule?\nFiles:\n- 文件.pdf\n", "English"),
        ("请问保费回赠是什么？", "Chinese (Simplified)"),
    ],
)
def test_plan_generator_includes_output_language_policy(question, expected):
    from core.deepsearch.plan.generator import PlanGenerator, PlannerSettings

    class _StubLLM:
        def __init__(self):
            self.last_messages = None

        def chat(self, messages, **kwargs):  # noqa: ANN001
            self.last_messages = messages
            return "[]"

    llm = _StubLLM()
    settings = PlannerSettings(
        mode="react",
        max_steps=3,
        enable_sub_question=False,
        system_prompt="SYS",
        user_prompt_template="Question: {question}\nOutput language: {output_language}\n{available_tools}",
        available_tools_hint="tools",
    )
    gen = PlanGenerator(llm, settings)
    with pytest.raises(RuntimeError):
        gen.generate_plan(question)
    assert llm.last_messages is not None
    assert f"Output language: {expected}" in llm.last_messages[1]["content"]


def test_infer_user_language_ignores_split_filename_cjk_prefix():
    from core.deepsearch.utils.language_policy import infer_user_language

    # Common pattern: CJK filename split across tokens, ending in ".pdf".
    q = "Answer in English. Document: AXA安盛「盛利II-至尊」保费回赠及预缴利率 截止至12月31日（英文版）.pdf"
    assert infer_user_language(q) == "en"
