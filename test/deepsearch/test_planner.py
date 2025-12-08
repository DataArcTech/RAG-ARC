"""DeepSearch planner tests: config wiring + real LLM plan generation."""
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import pytest
from dotenv import load_dotenv

load_dotenv()

pytest.importorskip("mcp", reason="DeepSearch planner tests require MCP client dependencies")

from config.core.deepsearch.plan_config import DeepSearchPlannerConfig
from core.deepsearch.plan import DeepSearchPlanner


def get_deepsearch_config() -> Dict[str, DeepSearchPlannerConfig]:
    """Return DeepSearch-related configs (planner only for now)."""

    return {"planner": DeepSearchPlannerConfig()}


def _ensure_chat_credentials() -> None:
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
        "text_channel_tool": "llm.summarize",
        "web_channel_tool": "web.search",
        "default_web_provider": os.getenv("DEEPSEARCH_WEB_PROVIDER") or "serper",
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
        result = asyncio.run(planner.build_plan(question, owner_id="planner-test-owner"))

        plan = result["plan"]
        assert plan["question"] == question
        assert plan["mode"] == plan_generator.settings.mode
        assert len(plan["steps"]) > 0

        artifact_path = result["artifact_path"]
        assert artifact_path is not None
        artifact_path = Path(artifact_path)
        assert artifact_path.is_file()

        saved_payload = json.loads(artifact_path.read_text())
        assert saved_payload["plan_id"] == result["plan_id"]
        assert saved_payload["question"] == question
        assert saved_payload["owner_id"] == "planner-test-owner"
    finally:
        shutil.rmtree(plan_output_dir, ignore_errors=True)
