import json
from types import SimpleNamespace

from core.deepsearch.trace import TraceEvent
from api.routers.rag_inference_modules.stream_chat.deepsearch.deepsearch_handler import (
    _build_progress_info,
    _build_trace_message,
)


def test_build_trace_message_prefers_llm_thinking_for_think_event() -> None:
    trace_event = TraceEvent(
        tag="think",
        content=json.dumps(
            {
                "thinking": "先确认公司主体，再核对保费口径。",
                "answer": {"tool_calls": []},
            },
            ensure_ascii=False,
        ),
        meta={"stage": "think_init"},
    )

    message = _build_trace_message(trace_event)

    assert message == "先确认公司主体，再核对保费口径。"


def test_build_trace_message_prefers_tool_response_summary_thinking() -> None:
    tool_response_payload = {
        "tool_name": "think",
        "result": {
            "summary": json.dumps(
                {
                    "thinking": "需要先查公司英文名是否对应同一实体。",
                    "answer": {"tool_calls": []},
                },
                ensure_ascii=False,
            ),
        },
    }
    trace_event = TraceEvent(
        tag="tool_response",
        content=json.dumps(tool_response_payload, ensure_ascii=False),
        meta={"tool_name": "think", "ok": True},
    )

    message = _build_trace_message(trace_event)

    assert message == "需要先查公司英文名是否对应同一实体。"


def test_build_progress_info_uses_latest_think_note_reasoning() -> None:
    state = SimpleNamespace(
        reasoning_trace={
            "think_notes": [
                {"reasoning": "通过 web.search 扩展候选来源，然后再交叉验证。"},
            ],
            "reasoning_steps": [],
            "tool_results": [],
        }
    )

    payload = _build_progress_info("reasoned", {}, state, request_id="req-1")

    assert payload["message"] == "通过 web.search 扩展候选来源，然后再交叉验证。"


def test_build_progress_info_falls_back_to_stage_name() -> None:
    state = SimpleNamespace()

    payload = _build_progress_info("created", {}, state, request_id="req-1")

    assert payload["message"] == "created"
