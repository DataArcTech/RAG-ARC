#!/usr/bin/env python3
"""简单的 DeepSearch trace 存储和查询功能测试脚本（可直接运行）"""
import os
import sys
import json
import uuid
import tempfile
import shutil
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

try:
    from core.deepsearch.trace import TraceEvent
    from api.routers.rag_inference_modules.stream_chat.deepsearch.deepsearch_handler import (
        _save_trace_events_to_file,
        load_trace_events_from_file,
    )
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保在正确的 Python 环境中运行此脚本")
    sys.exit(1)


def test_save_and_load_trace():
    """测试保存和加载 trace 事件"""
    print("\n" + "="*60)
    print("测试 1: 保存和加载 trace 事件")
    print("="*60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_deepsearch_trace_")
    print(f"📁 临时目录: {temp_dir}")
    
    try:
        # 设置环境变量
        os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"] = temp_dir
        
        # 创建示例 trace 事件
        trace_events = [
            TraceEvent(
                tag="think",
                content="正在思考问题...",
                meta={"stage": "plan", "step_id": "step_1"}
            ),
            TraceEvent(
                tag="tool_call",
                content="调用图谱查询工具",
                meta={"tool_name": "graph_adapter.query", "plan_step": "plan_01"}
            ),
            TraceEvent(
                tag="tool_response",
                content="查询完成，获取到 5 条证据",
                meta={"tool_name": "graph_adapter.query", "ok": True, "evidence_count": 5}
            ),
        ]
        
        # 创建示例 DeepSearch 结果
        deepsearch_result = {
            "plan": {
                "plan_id": "test_plan_id",
                "question": "测试问题",
            },
            "report": {
                "answer": "这是测试答案",
            },
        }
        
        request_id = str(uuid.uuid4())
        run_id = "test_run_123"
        query = "测试查询"
        
        print(f"📝 准备保存 {len(trace_events)} 个 trace 事件")
        print(f"   request_id: {request_id}")
        print(f"   run_id: {run_id}")
        print(f"   query: {query}")
        
        # 保存 trace 事件
        saved_path = _save_trace_events_to_file(
            trace_events=trace_events,
            request_id=request_id,
            run_id=run_id,
            query=query,
            deepsearch_result=deepsearch_result
        )
        
        if saved_path is None:
            print("❌ 保存失败: 返回路径为 None")
            return False
        
        if not os.path.exists(saved_path):
            print(f"❌ 保存失败: 文件不存在 {saved_path}")
            return False
        
        print(f"✅ 保存成功: {saved_path}")
        
        # 验证文件内容
        with open(saved_path, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
        
        # 验证结构
        assert "metadata" in trace_data, "缺少 metadata"
        assert "deepsearch_result" in trace_data, "缺少 deepsearch_result"
        assert "trace_events" in trace_data, "缺少 trace_events"
        
        print("✅ 文件结构验证通过")
        
        # 验证 metadata
        assert trace_data["metadata"]["request_id"] == request_id
        assert trace_data["metadata"]["run_id"] == run_id
        assert trace_data["metadata"]["query"] == query
        assert trace_data["metadata"]["total_events"] == len(trace_events)
        
        print("✅ metadata 验证通过")
        
        # 验证 trace_events
        assert len(trace_data["trace_events"]) == len(trace_events)
        assert trace_data["trace_events"][0]["tag"] == "think"
        
        print("✅ trace_events 验证通过")
        
        # 加载 trace 事件
        print(f"\n📖 加载 trace 事件: {saved_path}")
        loaded_data = load_trace_events_from_file(saved_path)
        
        if loaded_data is None:
            print("❌ 加载失败: 返回数据为 None")
            return False
        
        print("✅ 加载成功")
        
        # 验证加载的数据
        assert loaded_data["metadata"]["request_id"] == request_id
        assert len(loaded_data["trace_events"]) == len(trace_events)
        
        print("✅ 加载数据验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理
        if "DEEPSEARCH_TRACE_STORAGE_PATH" in os.environ:
            del os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"]
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"🧹 清理临时目录: {temp_dir}")


def test_load_nonexistent_file():
    """测试加载不存在的文件"""
    print("\n" + "="*60)
    print("测试 2: 加载不存在的文件")
    print("="*60)
    
    nonexistent_path = "/nonexistent/path/trace_file.json"
    
    loaded_data = load_trace_events_from_file(nonexistent_path)
    
    if loaded_data is None:
        print("✅ 正确处理不存在的文件（返回 None）")
        return True
    else:
        print("❌ 应该返回 None，但返回了数据")
        return False


def test_trace_file_path_in_raw_llm_response():
    """测试 trace 文件路径在 raw_llm_response 中的存储逻辑"""
    print("\n" + "="*60)
    print("测试 3: trace 文件路径存储逻辑")
    print("="*60)
    
    deepsearch_trace_file_path = "/test/path/trace_file.json"
    
    # 测试场景 1: raw_llm_response 为字典
    raw_llm_response = {"response": "test response"}
    raw_llm_response_with_trace = raw_llm_response
    if deepsearch_trace_file_path:
        if raw_llm_response_with_trace is None:
            raw_llm_response_with_trace = {}
        if not isinstance(raw_llm_response_with_trace, dict):
            raw_llm_response_with_trace = {"response": raw_llm_response_with_trace}
        raw_llm_response_with_trace["deepsearch_trace_file_path"] = deepsearch_trace_file_path
    
    assert "deepsearch_trace_file_path" in raw_llm_response_with_trace
    assert raw_llm_response_with_trace["deepsearch_trace_file_path"] == deepsearch_trace_file_path
    print("✅ 场景 1: raw_llm_response 为字典 - 通过")
    
    # 测试场景 2: raw_llm_response 为 None
    raw_llm_response = None
    raw_llm_response_with_trace = raw_llm_response
    if deepsearch_trace_file_path:
        if raw_llm_response_with_trace is None:
            raw_llm_response_with_trace = {}
        if not isinstance(raw_llm_response_with_trace, dict):
            raw_llm_response_with_trace = {"response": raw_llm_response_with_trace}
        raw_llm_response_with_trace["deepsearch_trace_file_path"] = deepsearch_trace_file_path
    
    assert isinstance(raw_llm_response_with_trace, dict)
    assert "deepsearch_trace_file_path" in raw_llm_response_with_trace
    print("✅ 场景 2: raw_llm_response 为 None - 通过")
    
    # 测试场景 3: raw_llm_response 为字符串
    raw_llm_response = "test response string"
    raw_llm_response_with_trace = raw_llm_response
    if deepsearch_trace_file_path:
        if raw_llm_response_with_trace is None:
            raw_llm_response_with_trace = {}
        if not isinstance(raw_llm_response_with_trace, dict):
            raw_llm_response_with_trace = {"response": raw_llm_response_with_trace}
        raw_llm_response_with_trace["deepsearch_trace_file_path"] = deepsearch_trace_file_path
    
    assert isinstance(raw_llm_response_with_trace, dict)
    assert "deepsearch_trace_file_path" in raw_llm_response_with_trace
    assert raw_llm_response_with_trace["response"] == "test response string"
    print("✅ 场景 3: raw_llm_response 为字符串 - 通过")
    
    return True


def test_session_api_loading_logic():
    """测试 session API 中的 trace 加载逻辑"""
    print("\n" + "="*60)
    print("测试 4: session API trace 加载逻辑")
    print("="*60)
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="test_deepsearch_trace_")
    
    try:
        os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"] = temp_dir
        
        # 创建并保存 trace 事件
        trace_events = [
            TraceEvent(tag="think", content="测试", meta={})
        ]
        
        request_id = str(uuid.uuid4())
        saved_path = _save_trace_events_to_file(
            trace_events=trace_events,
            request_id=request_id,
            run_id="test_run",
            query="测试",
            deepsearch_result=None
        )
        
        # 模拟消息对象
        class MockMessage:
            def __init__(self, trace_path):
                self.raw_llm_response = {
                    "deepsearch_trace_file_path": trace_path
                }
        
        msg = MockMessage(saved_path)
        
        # 模拟 session.py 中的加载逻辑
        deepsearch_trace = None
        if hasattr(msg, "raw_llm_response") and msg.raw_llm_response:
            if isinstance(msg.raw_llm_response, dict):
                trace_file_path = msg.raw_llm_response.get("deepsearch_trace_file_path")
                if trace_file_path:
                    deepsearch_trace = load_trace_events_from_file(trace_file_path)
        
        if deepsearch_trace is None:
            print("❌ 加载失败: trace 数据为 None")
            return False
        
        assert "metadata" in deepsearch_trace
        assert "trace_events" in deepsearch_trace
        print("✅ session API 加载逻辑验证通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if "DEEPSEARCH_TRACE_STORAGE_PATH" in os.environ:
            del os.environ["DEEPSEARCH_TRACE_STORAGE_PATH"]
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("DeepSearch Trace 存储和查询功能测试")
    print("="*60)
    
    tests = [
        ("保存和加载 trace 事件", test_save_and_load_trace),
        ("加载不存在的文件", test_load_nonexistent_file),
        ("trace 文件路径存储逻辑", test_trace_file_path_in_raw_llm_response),
        ("session API 加载逻辑", test_session_api_loading_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 抛出异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
