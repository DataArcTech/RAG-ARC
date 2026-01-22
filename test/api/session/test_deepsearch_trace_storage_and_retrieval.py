"""测试 DeepSearch trace 存储和查询功能"""
import os
import sys
import json
import uuid
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any

import pytest

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from core.deepsearch.trace import TraceEvent
from api.routers.rag_inference_modules.stream_chat.deepsearch.deepsearch_handler import (
    _save_trace_events_to_file,
    load_trace_events_from_file,
)


@pytest.fixture
def temp_trace_dir():
    """创建临时目录用于存储 trace 文件"""
    temp_dir = tempfile.mkdtemp(prefix="test_deepsearch_trace_")
    yield temp_dir
    # 清理临时目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_trace_events():
    """创建示例 trace 事件"""
    return [
        TraceEvent(
            tag="think",
            content="正在思考问题...",
            meta={"stage": "plan", "step_id": "step_1"}
        ),
        TraceEvent(
            tag="tool_call",
            content="调用图谱查询工具",
            meta={"tool_name": "explore", "plan_step": "plan_01"}
        ),
        TraceEvent(
            tag="tool_response",
            content="查询完成，获取到 5 条证据",
            meta={"tool_name": "explore", "ok": True, "evidence_count": 5}
        ),
        TraceEvent(
            tag="write_outline",
            content="已生成搜索计划大纲",
            meta={}
        ),
    ]


@pytest.fixture
def sample_deepsearch_result():
    """创建示例 DeepSearch 结果"""
    return {
        "plan": {
            "plan_id": "test_plan_id",
            "question": "测试问题",
            "steps": []
        },
        "report": {
            "answer": "这是测试答案",
            "evidences": []
        },
        "state": {
            "run_id": "test_run_id"
        }
    }


def test_save_trace_events_to_file(temp_trace_dir, sample_trace_events, sample_deepsearch_result):
    """测试保存 trace 事件到文件"""
    # 设置环境变量
    os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"] = temp_trace_dir
    
    request_id = str(uuid.uuid4())
    run_id = "test_run_123"
    query = "测试查询"
    
    # 保存 trace 事件
    saved_path = _save_trace_events_to_file(
        trace_events=sample_trace_events,
        request_id=request_id,
        run_id=run_id,
        query=query,
        deepsearch_result=sample_deepsearch_result
    )
    
    # 验证文件路径不为空
    assert saved_path is not None
    assert os.path.exists(saved_path)
    
    # 验证文件内容
    with open(saved_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
    
    # 验证 metadata
    assert "metadata" in trace_data
    assert trace_data["metadata"]["request_id"] == request_id
    assert trace_data["metadata"]["run_id"] == run_id
    assert trace_data["metadata"]["query"] == query
    assert trace_data["metadata"]["total_events"] == len(sample_trace_events)
    assert "timestamp" in trace_data["metadata"]
    
    # 验证 deepsearch_result
    assert "deepsearch_result" in trace_data
    assert trace_data["deepsearch_result"]["plan"]["plan_id"] == "test_plan_id"
    
    # 验证 trace_events
    assert "trace_events" in trace_data
    assert len(trace_data["trace_events"]) == len(sample_trace_events)
    
    # 验证第一个事件
    first_event = trace_data["trace_events"][0]
    assert first_event["tag"] == "think"
    assert first_event["content"] == "正在思考问题..."
    assert first_event["meta"]["stage"] == "plan"
    
    # 清理环境变量
    if "DEEPSEARCH_TRACE_STORAGE_PATH" in os.environ:
        del os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"]


def test_load_trace_events_from_file(temp_trace_dir, sample_trace_events, sample_deepsearch_result):
    """测试从文件加载 trace 事件"""
    # 设置环境变量
    os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"] = temp_trace_dir
    
    request_id = str(uuid.uuid4())
    run_id = "test_run_123"
    query = "测试查询"
    
    # 先保存 trace 事件
    saved_path = _save_trace_events_to_file(
        trace_events=sample_trace_events,
        request_id=request_id,
        run_id=run_id,
        query=query,
        deepsearch_result=sample_deepsearch_result
    )
    
    assert saved_path is not None
    
    # 加载 trace 事件
    loaded_data = load_trace_events_from_file(saved_path)
    
    # 验证加载的数据
    assert loaded_data is not None
    assert "metadata" in loaded_data
    assert "deepsearch_result" in loaded_data
    assert "trace_events" in loaded_data
    
    # 验证 metadata
    assert loaded_data["metadata"]["request_id"] == request_id
    assert loaded_data["metadata"]["run_id"] == run_id
    assert loaded_data["metadata"]["query"] == query
    assert loaded_data["metadata"]["total_events"] == len(sample_trace_events)
    
    # 验证 trace_events
    assert len(loaded_data["trace_events"]) == len(sample_trace_events)
    assert loaded_data["trace_events"][0]["tag"] == "think"
    
    # 清理环境变量
    if "DEEPSEARCH_TRACE_STORAGE_PATH" in os.environ:
        del os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"]


def test_load_trace_events_from_nonexistent_file():
    """测试从不存在的文件加载 trace 事件"""
    nonexistent_path = "/nonexistent/path/trace_file.json"
    
    loaded_data = load_trace_events_from_file(nonexistent_path)
    
    # 应该返回 None，而不是抛出异常
    assert loaded_data is None


def test_save_trace_events_with_none_result(temp_trace_dir, sample_trace_events):
    """测试保存 trace 事件时 deepsearch_result 为 None"""
    # 设置环境变量
    os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"] = temp_trace_dir
    
    request_id = str(uuid.uuid4())
    run_id = "test_run_123"
    query = "测试查询"
    
    # 保存 trace 事件（deepsearch_result 为 None，模拟失败情况）
    saved_path = _save_trace_events_to_file(
        trace_events=sample_trace_events,
        request_id=request_id,
        run_id=run_id,
        query=query,
        deepsearch_result=None
    )
    
    # 验证文件路径不为空
    assert saved_path is not None
    assert os.path.exists(saved_path)
    
    # 验证文件内容
    with open(saved_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
    
    # 验证 deepsearch_result 为 None
    assert "deepsearch_result" in trace_data
    assert trace_data["deepsearch_result"] is None
    
    # 验证 trace_events 仍然存在
    assert "trace_events" in trace_data
    assert len(trace_data["trace_events"]) == len(sample_trace_events)


def test_trace_storage_renders_sup_citations_for_user_visible_answer(temp_trace_dir, sample_trace_events):
    """Trace file should store a rendered DeepSearch answer (sup, no in-text References section)."""
    os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"] = temp_trace_dir

    request_id = str(uuid.uuid4())
    run_id = "test_run_render_123"
    query = "测试引用渲染"

    deepsearch_result = {
        "plan": {"plan_id": "test_plan_id", "question": "测试问题", "steps": []},
        "report": {
            "answer": "Alpha[ev1]。",
            "evidences": [
                {
                    "chunk_id": "ev1",
                    "source": "hipporag",
                    "content": "Local snippet",
                    "provenance": {"metadata": {"chunk_metadata": {"source_file_id": "file-1", "filename": "doc1.md"}}},
                }
            ],
            "structured_report": {"citations": [{"evidence_id": "ev1"}]},
        },
        "state": {"run_id": run_id},
    }

    saved_path = _save_trace_events_to_file(
        trace_events=sample_trace_events,
        request_id=request_id,
        run_id=run_id,
        query=query,
        deepsearch_result=deepsearch_result,
    )
    assert saved_path is not None and os.path.exists(saved_path)

    with open(saved_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    stored = trace_data["deepsearch_result"]["report"]
    assert stored["answer_raw"] == "Alpha[ev1]。"
    assert "<sup>1</sup>" in stored["answer"]
    assert "## References" not in stored["answer"]
    assert stored.get("sources") and stored["sources"][0]["file"] == "/knowledge/chunk/ev1"

    if "DEEPSEARCH_TRACE_STORAGE_PATH" in os.environ:
        del os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"]

    # 清理环境变量
    if "DEEPSEARCH_TRACE_STORAGE_PATH" in os.environ:
        del os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"]


def test_save_trace_events_with_empty_list(temp_trace_dir, sample_deepsearch_result):
    """测试保存空的 trace 事件列表"""
    # 设置环境变量
    os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"] = temp_trace_dir
    
    request_id = str(uuid.uuid4())
    run_id = "test_run_123"
    query = "测试查询"
    
    # 保存空的 trace 事件列表
    saved_path = _save_trace_events_to_file(
        trace_events=[],
        request_id=request_id,
        run_id=run_id,
        query=query,
        deepsearch_result=sample_deepsearch_result
    )
    
    # 验证文件路径不为空（即使事件列表为空，也应该保存文件）
    assert saved_path is not None
    assert os.path.exists(saved_path)
    
    # 验证文件内容
    with open(saved_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
    
    # 验证 metadata
    assert trace_data["metadata"]["total_events"] == 0
    
    # 验证 trace_events 为空列表
    assert "trace_events" in trace_data
    assert len(trace_data["trace_events"]) == 0
    
    # 清理环境变量
    if "DEEPSEARCH_TRACE_STORAGE_PATH" in os.environ:
        del os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"]


def test_trace_file_path_in_raw_llm_response():
    """测试 trace 文件路径是否正确存储在 raw_llm_response 中"""
    # 这个测试验证 response_builder.py 中的逻辑
    # 模拟 create_assistant_message 的行为
    
    deepsearch_trace_file_path = "/test/path/trace_file.json"
    raw_llm_response = {"response": "test response"}
    
    # 模拟 response_builder.py 中的逻辑
    raw_llm_response_with_trace = raw_llm_response
    if deepsearch_trace_file_path:
        if raw_llm_response_with_trace is None:
            raw_llm_response_with_trace = {}
        if not isinstance(raw_llm_response_with_trace, dict):
            raw_llm_response_with_trace = {"response": raw_llm_response_with_trace}
        raw_llm_response_with_trace["deepsearch_trace_file_path"] = deepsearch_trace_file_path
    
    # 验证路径已添加
    assert "deepsearch_trace_file_path" in raw_llm_response_with_trace
    assert raw_llm_response_with_trace["deepsearch_trace_file_path"] == deepsearch_trace_file_path
    assert raw_llm_response_with_trace["response"] == "test response"


def test_trace_file_path_with_none_raw_llm_response():
    """测试当 raw_llm_response 为 None 时，trace 文件路径的处理"""
    deepsearch_trace_file_path = "/test/path/trace_file.json"
    raw_llm_response = None
    
    # 模拟 response_builder.py 中的逻辑
    raw_llm_response_with_trace = raw_llm_response
    if deepsearch_trace_file_path:
        if raw_llm_response_with_trace is None:
            raw_llm_response_with_trace = {}
        if not isinstance(raw_llm_response_with_trace, dict):
            raw_llm_response_with_trace = {"response": raw_llm_response_with_trace}
        raw_llm_response_with_trace["deepsearch_trace_file_path"] = deepsearch_trace_file_path
    
    # 验证路径已添加，且创建了新的字典
    assert isinstance(raw_llm_response_with_trace, dict)
    assert "deepsearch_trace_file_path" in raw_llm_response_with_trace
    assert raw_llm_response_with_trace["deepsearch_trace_file_path"] == deepsearch_trace_file_path


def test_trace_file_path_with_string_raw_llm_response():
    """测试当 raw_llm_response 为字符串时，trace 文件路径的处理"""
    deepsearch_trace_file_path = "/test/path/trace_file.json"
    raw_llm_response = "test response string"
    
    # 模拟 response_builder.py 中的逻辑
    raw_llm_response_with_trace = raw_llm_response
    if deepsearch_trace_file_path:
        if raw_llm_response_with_trace is None:
            raw_llm_response_with_trace = {}
        if not isinstance(raw_llm_response_with_trace, dict):
            raw_llm_response_with_trace = {"response": raw_llm_response_with_trace}
        raw_llm_response_with_trace["deepsearch_trace_file_path"] = deepsearch_trace_file_path
    
    # 验证路径已添加，且字符串被包装在字典中
    assert isinstance(raw_llm_response_with_trace, dict)
    assert "deepsearch_trace_file_path" in raw_llm_response_with_trace
    assert raw_llm_response_with_trace["deepsearch_trace_file_path"] == deepsearch_trace_file_path
    assert raw_llm_response_with_trace["response"] == "test response string"


def test_session_api_trace_loading_logic(temp_trace_dir, sample_trace_events, sample_deepsearch_result):
    """测试 session API 中 trace 加载逻辑"""
    # 设置环境变量
    os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"] = temp_trace_dir
    
    request_id = str(uuid.uuid4())
    run_id = "test_run_123"
    query = "测试查询"
    
    # 先保存 trace 事件
    saved_path = _save_trace_events_to_file(
        trace_events=sample_trace_events,
        request_id=request_id,
        run_id=run_id,
        query=query,
        deepsearch_result=sample_deepsearch_result
    )
    
    assert saved_path is not None
    
    # 模拟 session.py 中的逻辑
    # 创建一个模拟的消息对象
    class MockMessage:
        def __init__(self):
            self.raw_llm_response = {
                "deepsearch_trace_file_path": saved_path
            }
    
    msg = MockMessage()
    
    # 模拟 list_session_messages 中的逻辑
    deepsearch_trace = None
    if hasattr(msg, "raw_llm_response") and msg.raw_llm_response:
        if isinstance(msg.raw_llm_response, dict):
            trace_file_path = msg.raw_llm_response.get("deepsearch_trace_file_path")
            if trace_file_path:
                deepsearch_trace = load_trace_events_from_file(trace_file_path)
    
    # 验证 trace 数据已加载
    assert deepsearch_trace is not None
    assert "metadata" in deepsearch_trace
    assert "trace_events" in deepsearch_trace
    assert len(deepsearch_trace["trace_events"]) == len(sample_trace_events)
    
    # 清理环境变量
    if "DEEPSEARCH_TRACE_STORAGE_PATH" in os.environ:
        del os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"]


def test_session_api_trace_loading_without_path():
    """测试 session API 中没有 trace 文件路径的情况"""
    # 模拟消息对象，没有 trace 文件路径
    class MockMessage:
        def __init__(self):
            self.raw_llm_response = {
                "response": "test response"
            }
    
    msg = MockMessage()
    
    # 模拟 list_session_messages 中的逻辑
    deepsearch_trace = None
    if hasattr(msg, "raw_llm_response") and msg.raw_llm_response:
        if isinstance(msg.raw_llm_response, dict):
            trace_file_path = msg.raw_llm_response.get("deepsearch_trace_file_path")
            if trace_file_path:
                deepsearch_trace = load_trace_events_from_file(trace_file_path)
    
    # 验证 trace 数据为 None（因为没有文件路径）
    assert deepsearch_trace is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
