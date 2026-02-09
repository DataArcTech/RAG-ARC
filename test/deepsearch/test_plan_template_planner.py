import pytest

from core.deepsearch.planning import coerce_templates, instantiate_template_plan, select_plan_template


class _FakeLLM:
    def __init__(self, responses):  # noqa: ANN001
        self._responses = list(responses)

    async def achat(self, messages, **kwargs):  # noqa: ANN001, ARG002
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def test_instantiate_template_plan_renders_slots_and_question() -> None:
    templates = coerce_templates()
    plan_items, tool_calls, signature = instantiate_template_plan(
        templates=templates,
        template_id="doc.numeric_table.v1",
        question="Which term has the highest IRR?",
        slots={"metric": "IRR", "context": "premium term options"},
    )
    assert signature
    assert any("IRR" in item["text"] for item in plan_items)
    assert tool_calls and tool_calls[0]["tool_name"] == "explore"
    actions = tool_calls[0]["tool_args"]["actions"]
    assert actions[0]["tool"] == "search.file"
    assert actions[0]["args"]["query"] == "Which term has the highest IRR?"


@pytest.mark.asyncio
async def test_select_plan_template_accepts_known_template_id() -> None:
    llm = _FakeLLM(
        [
            '{"use_template": true, "template_id": "doc.compare.v1", "slots": {"targets": "A,B", "axis": "fees"}, "report_needed": true, "report_style": "deepsearch", "reasoning": "comparison"}'
        ]
    )
    selection = await select_plan_template(llm_connector=llm, question="Compare A vs B fees")
    assert selection is not None
    assert selection.use_template is True
    assert selection.template_id == "doc.compare.v1"
    assert selection.slots.get("axis") == "fees"


@pytest.mark.asyncio
async def test_select_plan_template_rejects_unknown_template_id() -> None:
    llm = _FakeLLM(
        [
            '{"use_template": true, "template_id": "doc.unknown.v9", "slots": {}, "report_needed": true, "report_style": "deepsearch", "reasoning": "x"}'
        ]
    )
    selection = await select_plan_template(llm_connector=llm, question="Anything")
    assert selection is not None
    assert selection.use_template is False
    assert selection.template_id is None
