import pytest

from encapsulation.data_model.deepsearch import EvidenceChunk, GraphQueryContext, ToolResultPayload
from core.deepsearch.gap import GapDetectionEngine, GapDetectionEvaluator, GapDetectionSettings
from encapsulation.deepsearch.external import ExternalSearchChannel


class _Telemetry:
    def __init__(self):
        self.events = []

    def log_gap_detection(self, *, result, context):
        self.events.append(("gap", result, context))

    def log_external_channel(self, payload):
        self.events.append(("external", payload))


class _StubToolManager:
    def __init__(self):
        self.calls = []

    async def invoke(self, tool_name: str, *, payload):
        self.calls.append((tool_name, payload))
        chunk = EvidenceChunk(
            chunk_id="tool-chunk-1",
            source=tool_name,
            content="external evidence",
        )
        return ToolResultPayload(
            tool_name=tool_name,
            namespace=f"stub::{tool_name}",
            channel="web",
            profile="X",
            determinism="stochastic",
            summary="stub summary",
            evidences=[chunk],
            diagnostics={"provider": "stub"},
            think_notes=[],
        )


def _gap_engine(telemetry=None, config=None):
    evaluator = GapDetectionEvaluator(GapDetectionSettings(expected_min_chunks=3, coverage_threshold=0.9))
    return GapDetectionEngine(evaluator, telemetry_client=telemetry, config=config)


def _reasoning_trace():
    context = GraphQueryContext(adapter_name="hipporag", question="Who founded OpenAI?")
    return {
        "question": context.question,
        "graph_context": context.model_dump(exclude_none=True),
        "evidences": [
            {"chunk_id": "c1", "source": "graph", "content": "OpenAI founded by Sam Altman"},
        ],
        "coverage_metrics": {"answer_confidence": 0.3},
        "pending_external": [
            {
                "step_id": "plan_ext",
                "description": "Query web",
                "channel": "web",
                "tool": "web.search",
                "metadata": {"query": "Who founded OpenAI?"},
            }
        ],
    }


def test_gap_detection_respects_env_disable(monkeypatch):
    telemetry = _Telemetry()
    engine = _gap_engine(telemetry=telemetry, config={"enable_external_on_gap": True})
    monkeypatch.setenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", "false")

    result = engine.evaluate(_reasoning_trace())

    assert result["should_trigger_external"] is False
    assert result["reason"] == "external_disabled"
    assert telemetry.events and telemetry.events[0][0] == "gap"
    monkeypatch.delenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", raising=False)


def test_gap_detection_triggered_by_think_note(monkeypatch):
    telemetry = _Telemetry()
    engine = _gap_engine(telemetry=telemetry, config={"enable_external_on_gap": True})
    trace = _reasoning_trace()
    trace["coverage_metrics"] = {"answer_confidence": 0.95, "coverage_ratio": 0.9}
    trace["think_notes"] = [
        {
            "plan_step_id": "think_auto_01",
            "reasoning": "Need external research",
            "metadata": {"gap_trigger": True, "missing_topics": ["financials"]},
        }
    ]

    result = engine.evaluate(trace)

    assert result["should_trigger_external"] is True
    assert result["reason"] == "think_gap"
    assert "financials" in result["missing_topics"]
    assert telemetry.events and telemetry.events[-1][0] == "gap"


def test_gap_detection_think_gap_respects_env_disable(monkeypatch):
    engine = _gap_engine(telemetry=None, config={"enable_external_on_gap": True})
    trace = _reasoning_trace()
    trace["coverage_metrics"] = {"answer_confidence": 0.95, "coverage_ratio": 0.9}
    trace["think_notes"] = [
        {
            "plan_step_id": "think_auto_01",
            "reasoning": "Need external research",
            "metadata": {"gap_trigger": True, "missing_topics": ["financials"]},
        }
    ]
    monkeypatch.setenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", "false")

    result = engine.evaluate(trace)

    assert result["should_trigger_external"] is False
    assert result["reason"] == "external_disabled"
    diagnostics = result.get("diagnostics") or {}
    assert diagnostics.get("external_allowed") is False
    assert diagnostics.get("think_gap_trigger") is True
    assert diagnostics.get("think_gap_blocked") is True
    monkeypatch.delenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", raising=False)


@pytest.mark.asyncio
async def test_external_channel_calls_tool_manager(monkeypatch):
    telemetry = _Telemetry()
    tool_manager = _StubToolManager()
    channel = ExternalSearchChannel(
        tool_manager=tool_manager,
        config={"enabled": True, "default_provider": "stub"},
        telemetry_client=telemetry,
    )
    trace = _reasoning_trace()
    monkeypatch.setenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", "true")

    payload = await channel.run(trace["pending_external"], reasoning_trace=trace, gap_result={"reason": "coverage"})
    evidences = payload["evidences"]
    logs = payload["logs"]
    assert evidences and evidences[0]["source"] == "web.search"
    assert tool_manager.calls and tool_manager.calls[0][0] == "web.search"
    assert telemetry.events and telemetry.events[-1][0] == "external"
    assert logs and logs[0]["status"] == "ok"
    monkeypatch.delenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", raising=False)


@pytest.mark.asyncio
async def test_external_channel_respects_env_off(monkeypatch):
    tool_manager = _StubToolManager()
    channel = ExternalSearchChannel(tool_manager=tool_manager, config={"enabled": True})
    trace = _reasoning_trace()
    monkeypatch.setenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", "false")

    payload = await channel.run(trace["pending_external"], reasoning_trace=trace, gap_result=None)

    assert payload["evidences"] == []
    assert payload["logs"] == []
    assert not tool_manager.calls
    monkeypatch.delenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", raising=False)


@pytest.mark.asyncio
async def test_external_channel_falls_back_to_provider(monkeypatch):
    channel = ExternalSearchChannel(tool_manager=None, config={"enabled": True, "default_provider": "tavily"})
    trace = _reasoning_trace()
    monkeypatch.setenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", "true")

    async def _fake_tavily(self, task, *, question):
        return (
            [
                {
                    "chunk_id": "tavily-plan_ext-0",
                    "source": "web.tavily",
                    "content": "Result snippet",
                    "provenance": {"provider": "tavily"},
                }
            ],
            {"result_count": 1},
        )

    monkeypatch.setattr(ExternalSearchChannel, "_execute_with_tavily", _fake_tavily, raising=False)

    payload = await channel.run(trace["pending_external"], reasoning_trace=trace, gap_result=None)

    assert payload["evidences"] and payload["evidences"][0]["source"] == "web.tavily"
    monkeypatch.delenv("DEEPSEARCH_EXTERNAL_SEARCH_ENABLED", raising=False)
