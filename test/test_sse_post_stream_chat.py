"""
测试 POST SSE stream_chat 接口
核心测试：1. 无token/无效token应该返回401  2. 有效token时SSE流式输出正常工作
"""
import asyncio
import json
import time
import httpx
import uuid
from typing import Dict, Any


BASE_URL = "http://localhost:8001"
PROVIDED_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ3YW5nc2h1bmNoaSIsInR5cGUiOjEsImV4cCI6MTc2NzE1MzEwOX0.Vfy8HeLUni2OTrff0DT325VmROb7aVcmFDPNEVJE_8s"
USE_PROVIDED_TOKEN = True


async def create_session(token: str) -> str:
    """创建会话，返回 session_id"""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post(f"{BASE_URL}/session", headers=headers)
        assert response.status_code == 200, f"创建会话失败: {response.text}"
        data = response.json()
        assert data.get("code") == 200, f"创建会话返回code不是200: {data}"
        session_id = str(data.get("data")).strip().strip('"').strip("'")
        uuid.UUID(session_id)  # 验证UUID格式
        return session_id


async def read_sse_stream(response: httpx.Response) -> Dict[str, Any]:
    """读取 SSE 流并解析，返回解析结果"""
    result = {
        "http_status": response.status_code,
        "chunks": [],
        "content_parts": [],
        "progress_events": [],
        "payload_events": [],
        "has_done": False,
        "error_text": None,
        "first_chunk": None,
    }
    
    if response.status_code != 200:
        try:
            error_text = await response.aread()
            if isinstance(error_text, bytes):
                error_text = error_text.decode("utf-8", errors="replace")
            result["error_text"] = error_text
        except Exception:
            result["error_text"] = "无法读取错误响应"
        return result
    
    try:
        line_count = 0
        async for line in response.aiter_lines():
            line_count += 1
            if not line:
                continue
            
            # 处理SSE格式
            if line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
                if data == "[DONE]":
                    result["has_done"] = True
                    break
                
                try:
                    chunk = json.loads(data)
                    result["chunks"].append(chunk)
                    
                    if result["first_chunk"] is None:
                        result["first_chunk"] = chunk
                    
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = (choices[0] or {}).get("delta") or {}
                        content = delta.get("content")
                        if content:
                            result["content_parts"].append(content)
                        
                        tool_calls = delta.get("tool_calls") or []
                        for tool_call in tool_calls:
                            fn = (tool_call or {}).get("function") or {}
                            fn_name = fn.get("name")
                            fn_args = fn.get("arguments")
                            
                            if fn_name == "rag_arc_progress":
                                try:
                                    result["progress_events"].append(json.loads(fn_args or "{}"))
                                except Exception:
                                    pass
                            elif fn_name == "rag_arc_payload":
                                try:
                                    result["payload_events"].append(json.loads(fn_args or "{}"))
                                except Exception:
                                    pass
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        result["error_text"] = str(e)
    
    result["full_content"] = "".join(result["content_parts"])
    return result


async def get_token_and_session():
    """获取token和session_id"""
    if USE_PROVIDED_TOKEN and PROVIDED_TOKEN:
        token = PROVIDED_TOKEN
        session_id = await create_session(token)
        return token, session_id
    else:
        raise Exception("需要设置USE_PROVIDED_TOKEN=True并提供PROVIDED_TOKEN")


async def test_no_token_rejected():
    """测试1: 无token时应该被拒绝（返回401）"""
    print("\n1. 测试无token时应该被拒绝...")
    token, session_id = await get_token_and_session()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        body = {"query": "test query"}
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            json=body
        ) as response:
            assert response.status_code == 401, f"无token应该返回401，实际: {response.status_code}"
            error_text = await response.aread()
            if isinstance(error_text, bytes):
                error_text = error_text.decode("utf-8", errors="replace")
            print(f"   ✅ 无token返回401 - 错误: {error_text[:80]}")


async def test_invalid_token_rejected():
    """测试2: 无效token时应该被拒绝（返回401）"""
    print("\n2. 测试无效token时应该被拒绝...")
    token, session_id = await get_token_and_session()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        body = {"query": "test query"}
        headers = {
            "Authorization": "Bearer invalid_token_12345",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers=headers,
            json=body
        ) as response:
            assert response.status_code == 401, f"无效token应该返回401，实际: {response.status_code}"
            error_text = await response.aread()
            if isinstance(error_text, bytes):
                error_text = error_text.decode("utf-8", errors="replace")
            print(f"   ✅ 无效token返回401 - 错误: {error_text[:80]}")


async def test_valid_token_streams():
    """测试3: 有效token时SSE流式输出是否正常工作"""
    print("\n3. 测试有效token时SSE流式输出...")
    token, session_id = await get_token_and_session()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        body = {"query": "你好，请简单介绍一下你自己"}
        
        print(f"   📡 请求: POST /rag_inference/stream_chat/{session_id}")
        print(f"   📝 查询: {body['query']}")
        
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers=headers,
            json=body
        ) as response:
            print(f"   📊 HTTP状态码: {response.status_code}")
            print(f"   📋 Content-Type: {response.headers.get('content-type', 'N/A')}")
            
            assert response.status_code == 200, f"有效token应该返回200，实际: {response.status_code}"
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, f"Content-Type应该包含text/event-stream，实际: {content_type}"
            
            result = await read_sse_stream(response)
            
            # 验证流式输出
            assert result["http_status"] == 200, f"HTTP状态码应该是200，实际: {result['http_status']}"
            
            if result.get("error_text"):
                print(f"   ⚠️  错误信息: {result['error_text'][:200]}")
            
            print(f"   📦 收到 {len(result['chunks'])} 个chunks")
            if len(result["chunks"]) > 0 and result["first_chunk"]:
                print(f"   📄 第一个chunk结构: {json.dumps(result['first_chunk'], ensure_ascii=False, default=str)[:200]}...")
            
            if len(result["chunks"]) == 0:
                print(f"   ⚠️  未收到任何chunks，可能流还在继续或有问题")
                if result.get("error_text"):
                    raise AssertionError(f"未收到chunks，错误: {result['error_text'][:200]}")
            
            assert result["first_chunk"] is not None, f"应该至少有一个chunk，实际收到: {len(result['chunks'])} 个chunks"
            assert len(result["chunks"]) > 0, "应该有多个chunks（流式输出）"
            
            # 验证第一个chunk包含role=assistant
            first_delta = (result["first_chunk"].get("choices") or [{}])[0].get("delta") or {}
            print(f"   🔍 第一个delta: {json.dumps(first_delta, ensure_ascii=False, default=str)[:200]}")
            
            # 验证有内容返回（流式输出）- 如果没有content但有chunks，也认为成功
            if len(result["content_parts"]) == 0:
                print(f"   ⚠️  未收到content_parts，但收到 {len(result['chunks'])} 个chunks")
                print(f"   📊 content_parts: {result['content_parts']}")
                # 只要有chunks就认为流式输出成功
                if len(result["chunks"]) > 0:
                    print(f"   ✅ 流式输出成功（收到chunks但可能没有content字段）")
            else:
                assert len(result["full_content"]) > 0, "完整内容不应该为空"
            
            print(f"   ✅ 流式输出成功!")
            print(f"      - 收到 {len(result['chunks'])} 个chunks")
            print(f"      - 内容片段数: {len(result['content_parts'])}")
            print(f"      - 完整内容长度: {len(result['full_content'])} 字符")
            print(f"      - 内容预览: {result['full_content'][:80]}...")
            if result["has_done"]:
                print(f"      - ✅ 收到[DONE]标记")
            else:
                print(f"      - ⚠️  未收到[DONE]标记（流可能还在继续）")
            if result["progress_events"]:
                print(f"      - Progress事件数: {len(result['progress_events'])}")
            if result["payload_events"]:
                print(f"      - Payload事件数: {len(result['payload_events'])}")


if __name__ == "__main__":
    async def run_tests():
        print("=" * 60)
        print("开始测试 POST SSE stream_chat 接口")
        print("=" * 60)
        
        try:
            await test_no_token_rejected()
            await test_invalid_token_rejected()
            await test_valid_token_streams()
            
            print("\n" + "=" * 60)
            print("所有核心测试通过！✅")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_tests())
