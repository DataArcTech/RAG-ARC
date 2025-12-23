"""DeepSearch planner tests: config wiring + real LLM plan generation."""
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import pytest
from dotenv import load_dotenv

pytest.importorskip("mcp", reason="DeepSearch planner tests require MCP client dependencies")

from config.core.deepsearch.plan_config import DeepSearchPlannerConfig
from core.graph_adapter.base import GraphAccessScope
from core.deepsearch.plan import DeepSearchPlanner


def get_deepsearch_config() -> Dict[str, DeepSearchPlannerConfig]:
    """Return DeepSearch-related configs (planner only for now)."""

    return {"planner": DeepSearchPlannerConfig()}


def _ensure_chat_credentials() -> None:
    load_dotenv()
    if os.getenv("DEEPSEARCH_RUN_LLM_INTEGRATION_TESTS", "").strip().lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("Set DEEPSEARCH_RUN_LLM_INTEGRATION_TESTS=true to run planner LLM integration tests.")
    if not (
        os.getenv("CHAT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
    ):
        pytest.skip("Missing chat provider credentials; populate .env to run planner integration test.")


def test_get_deepsearch_config(monkeypatch):
    """Planner config should honor .env overrides and allow LLM disable."""

    monkeypatch.setenv("DEEPSEARCH_PLANNER_DISABLE_LLM", "true")
    monkeypatch.setenv("DEEPSEARCH_PLANNER_MODE", "iter_research")
    monkeypatch.setenv("DEEPSEARCH_PLANNER_MAX_STEPS", "4")
    monkeypatch.setenv("DEEPSEARCH_PLANNER_ENABLE_SUBQUESTION", "false")

    config = get_deepsearch_config()
    planner_cfg = config["planner"]

    plan_generator = planner_cfg.build()

    assert plan_generator.llm is None
    assert plan_generator.settings.mode == "iter_research"
    assert plan_generator.settings.max_steps == 4
    assert plan_generator.settings.enable_sub_question is False


def test_planner_generates_real_plan():
    """Run planner against a real LLM (requires chat credentials in .env)."""

    _ensure_chat_credentials()

    config = get_deepsearch_config()
    planner_cfg = config["planner"]
    plan_generator = planner_cfg.build()

    if plan_generator.llm is None:
        pytest.skip("Planner LLM is disabled via .env; enable it to run integration test.")

    plan_output_dir = Path("local") / "planner_test_runs"
    shutil.rmtree(plan_output_dir, ignore_errors=True)
    plan_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config = {
        "mode": plan_generator.settings.mode,
        "max_steps": plan_generator.settings.max_steps,
        "enable_sub_question": plan_generator.settings.enable_sub_question,
        "persist_plan": True,
        "plan_output_dir": str(plan_output_dir),
        "allow_external_channel": True,
        "graph_channel_tool": "graph_adapter.query",
        "text_channel_tool": "graph.context_rollup",
        "web_channel_tool": "web.search",
        "default_web_provider": os.getenv("DEEPSEARCH_WEB_PROVIDER") or "tavily",
        "graph_adapter_name": os.getenv("DEEPSEARCH_DEFAULT_ADAPTER") or "hipporag",
        "tool_arg_templates": {},
    }

    planner = DeepSearchPlanner(
        prompt_store=None,
        llm_connector=plan_generator.llm,
        config=runtime_config,
        plan_generator=plan_generator,
    )

    question = "RAG-ARC 是一个图谱优先的 Agent 框架，请输出发布计划调研步骤"

    try:
        result = asyncio.run(planner.build_plan(question, access_scope=GraphAccessScope(scope_id="planner-test")))

        plan = result["plan"]
        assert plan["question"] == question
        assert plan["mode"] == plan_generator.settings.mode
        assert len(plan["steps"]) > 0
        graph_context = plan.get("graph_context")
        assert graph_context
        assert graph_context["question"] == question
        assert graph_context["adapter_name"] == runtime_config["graph_adapter_name"]

        artifact_path = result["artifact_path"]
        assert artifact_path is not None
        artifact_path = Path(artifact_path)
        assert artifact_path.is_file()

        saved_payload = json.loads(artifact_path.read_text())
        assert saved_payload["plan_id"] == result["plan_id"]
        assert saved_payload["question"] == question
        assert saved_payload["graph_context"]["adapter_name"] == runtime_config["graph_adapter_name"]
    finally:
        shutil.rmtree(plan_output_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_planner_adapts_step_budget(tmp_path):
    runtime_config = {
        "mode": "react",
        "max_steps": 6,
        "enable_sub_question": True,
        "persist_plan": False,
        "plan_output_dir": str(tmp_path),
        "allow_external_channel": False,
        "graph_channel_tool": "graph_adapter.query",
        "text_channel_tool": "graph.context_rollup",
        "web_channel_tool": "web.search",
    }
    planner = DeepSearchPlanner(
        prompt_store=None,
        llm_connector=None,
        config=runtime_config,
    )
    scope = GraphAccessScope(scope_id="dynamic-test")

    short = await planner.build_plan("定义 RAG-ARC", access_scope=scope)
    complex_question = (
        "Compare RAG-ARC adoption plans across APAC, EMEA, and AMER, "
        "highlight blockers, timelines, and propose mitigation roadmap."
    )
    complex_plan = await planner.build_plan(complex_question, access_scope=scope)

    short_steps = short["plan"]["config"]["max_steps"]
    complex_steps = complex_plan["plan"]["config"]["max_steps"]

    assert short_steps < planner._base_max_steps
    assert complex_steps > short_steps
