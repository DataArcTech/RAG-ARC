"""Test that template planner is integrated into initial_think flow."""
from config.core.deepsearch.plan_templates import PLAN_TEMPLATES
from core.deepsearch.planning.template_planner import coerce_templates, instantiate_template_plan


def test_all_templates_instantiate_without_error() -> None:
    templates = coerce_templates()
    for tid, tmpl in templates.items():
        slots = {k: f"test_{k}" for k in tmpl.slots}
        plan_items, tool_calls, sig = instantiate_template_plan(
            templates=templates,
            template_id=tid,
            question="Test question?",
            slots=slots,
        )
        assert len(plan_items) >= 1
        assert sig


def test_default_plan_items_fallback_has_minimum_steps() -> None:
    from application.rag_inference.deepsearch.service_runtime.initial_think import _DEFAULT_PLAN_ITEMS
    assert len(_DEFAULT_PLAN_ITEMS) >= 3
    assert all(isinstance(item, dict) and "text" in item for item in _DEFAULT_PLAN_ITEMS)
