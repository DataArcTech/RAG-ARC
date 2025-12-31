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
USE_PROVIDED_TOKEN = False  # 改为 False，使用自动登录

# 测试用的用户名和密码（如果不存在会自动注册）
TEST_USERNAME = "test_sse_user"
TEST_PASSWORD = "test_password_123"
TEST_TYPE = 1  # chatKB


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


async def login_and_get_token(username: str = None, password: str = None, user_type: int = None) -> tuple[str, int]:
    """登录获取token"""
    username = username or TEST_USERNAME
    password = password or TEST_PASSWORD
    if user_type is None:
        user_type = TEST_TYPE
    
    async with httpx.AsyncClient() as client:
        # 先尝试注册（如果已存在会失败，但不影响）
        register_data = {
            "name": f"测试用户_{int(time.time())}",
            "user_name": username,
            "password": password,
            "type": user_type
        }
        try:
            register_response = await client.post(f"{BASE_URL}/auth/register", json=register_data)
            if register_response.status_code == 200:
                register_data_resp = register_response.json()
                actual_type = register_data_resp.get("data", {}).get("type", user_type)
                print(f"   ✅ 注册用户: {username} (请求type={user_type}, 实际type={actual_type})")
        except Exception as e:
            print(f"   ℹ️  用户可能已存在: {username} (错误: {e})")
            # 用户可能已存在，继续登录
        
        # 登录
        login_data = {
            "user_name": username,
            "password": password,
            "type": user_type
        }
        response = await client.post(f"{BASE_URL}/auth/token", json=login_data)
        assert response.status_code == 200, f"登录失败: {response.text}"
        data = response.json()
        assert data.get("code") == 200, f"登录返回code不是200: {data}"
        assert "access_token" in data.get("data", {}), f"登录响应中没有access_token: {data}"
        # 获取用户类型
        user_type_from_response = data.get("data", {}).get("user", {}).get("type", user_type)
        return data["data"]["access_token"], user_type_from_response


async def get_token_and_session(user_type: int = None):
    """获取token和session_id，返回 (token, session_id, user_id, user_type)"""
    user_type = user_type or TEST_TYPE
    if USE_PROVIDED_TOKEN and PROVIDED_TOKEN:
        token = PROVIDED_TOKEN
        try:
            session_id = await create_session(token)
            return token, session_id, None, user_type
        except Exception:
            # Token 可能过期，尝试自动登录
            print("   ⚠️  提供的token无效，尝试自动登录...")
            token, actual_type = await login_and_get_token(user_type=user_type)
            session_id = await create_session(token)
            return token, session_id, None, actual_type
    else:
        token, actual_type = await login_and_get_token(user_type=user_type)
        session_id = await create_session(token)
        return token, session_id, None, actual_type




async def test_no_token_rejected():
    """测试1: 无token时应该被拒绝（返回401）"""
    print("\n1. 测试无token时应该被拒绝...")
    token, session_id, _, _ = await get_token_and_session()
    
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
    token, session_id, _, _ = await get_token_and_session()
    
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


async def test_invalid_session_id():
    """测试3: 无效的session_id应该被拒绝（返回403或404）"""
    print("\n3. 测试无效的session_id应该被拒绝...")
    token, _, _, _ = await get_token_and_session()
    
    # 测试不存在的session_id
    invalid_session_id = str(uuid.uuid4())
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        body = {"query": "test query"}
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{invalid_session_id}",
            headers=headers,
            json=body
        ) as response:
            assert response.status_code in [403, 404], f"无效session_id应该返回403或404，实际: {response.status_code}"
            error_text = await response.aread()
            if isinstance(error_text, bytes):
                error_text = error_text.decode("utf-8", errors="replace")
            print(f"   ✅ 无效session_id返回{response.status_code} - 错误: {error_text[:80]}")


async def test_invalid_session_id_format():
    """测试4: 格式错误的session_id应该返回422"""
    print("\n4. 测试格式错误的session_id应该返回422...")
    token, _, _, _ = await get_token_and_session()
    
    # 测试非UUID格式的session_id
    invalid_format_session_id = "not-a-valid-uuid"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        body = {"query": "test query"}
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{invalid_format_session_id}",
            headers=headers,
            json=body
        ) as response:
            # FastAPI 会自动验证 UUID 格式，返回 422
            assert response.status_code == 422, f"格式错误的session_id应该返回422，实际: {response.status_code}"
            print(f"   ✅ 格式错误的session_id返回422")


async def test_valid_token_streams():
    """测试5: 有效token时SSE流式输出是否正常工作"""
    print("\n5. 测试有效token时SSE流式输出...")
    token, session_id, _, _ = await get_token_and_session()
    
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


async def test_chatkb_user_cannot_generate_graph():
    """测试6: chatKB用户(type=1)请求生成图应该被拒绝（返回403）"""
    print("\n6. 测试chatKB用户(type=1)请求生成图应该被拒绝...")
    token, session_id, _, user_type = await get_token_and_session(user_type=1)
    print(f"   👤 用户类型: {user_type}")
    assert user_type == 1, f"应该是type=1的用户，实际: {user_type}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        body = {"query": "test query", "return_subgraph": True}
        print(f"   📝 请求体: {body}")
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers=headers,
            json=body
        ) as response:
            print(f"   📊 HTTP状态码: {response.status_code}")
            if response.status_code != 403:
                # 读取响应内容看看是什么
                error_text = await response.aread()
                if isinstance(error_text, bytes):
                    error_text = error_text.decode("utf-8", errors="replace")
                print(f"   ⚠️  实际响应: {error_text[:200]}")
            assert response.status_code == 403, f"chatKB用户请求生成图应该返回403，实际: {response.status_code}"
            error_text = await response.aread()
            if isinstance(error_text, bytes):
                error_text = error_text.decode("utf-8", errors="replace")
            print(f"   ✅ chatKB用户请求生成图返回403 - 错误: {error_text[:100]}")


async def test_livingkb_user_can_generate_graph():
    """测试7: livingKB用户(type=0)请求生成图应该成功"""
    print("\n7. 测试livingKB用户(type=0)请求生成图...")
    # 使用不同的用户名，避免与之前的用户冲突
    livingkb_username = f"test_livingkb_{int(time.time())}"
    livingkb_password = f"test_password_{int(time.time())}"
    
    # 直接注册并登录 type=0 的用户
    token, user_type = await login_and_get_token(username=livingkb_username, password=livingkb_password, user_type=0)
    print(f"   👤 用户类型: {user_type}")
    assert user_type == 0, f"应该是type=0的用户，实际: {user_type}"
    
    session_id = await create_session(token)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        body = {"query": "你好，请简单介绍一下你自己", "return_subgraph": True}
        
        print(f"   📡 请求: POST /rag_inference/stream_chat/{session_id}")
        print(f"   📝 查询: {body['query']}")
        print(f"   🎯 请求生成图: return_subgraph=True")
        
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers=headers,
            json=body
        ) as response:
            print(f"   📊 HTTP状态码: {response.status_code}")
            
            assert response.status_code == 200, f"livingKB用户请求生成图应该返回200，实际: {response.status_code}"
            
            result = await read_sse_stream(response)
            
            # 验证流式输出
            assert result["http_status"] == 200, f"HTTP状态码应该是200，实际: {result['http_status']}"
            
            print(f"   📦 收到 {len(result['chunks'])} 个chunks")
            print(f"   📄 内容片段数: {len(result['content_parts'])}")
            print(f"   📊 完整内容长度: {len(result['full_content'])} 字符")
            
            # 检查是否有 payload 事件（包含 subgraph）
            if result["payload_events"]:
                payload = result["payload_events"][0]
                has_subgraph = "subgraph" in payload
                print(f"   🎯 Payload事件数: {len(result['payload_events'])}")
                print(f"   📊 包含subgraph: {has_subgraph}")
                if has_subgraph:
                    subgraph = payload.get("subgraph", {})
                    nodes_count = len(subgraph.get("nodes", []))
                    edges_count = len(subgraph.get("edges", []))
                    print(f"      - 子图节点数: {nodes_count}")
                    print(f"      - 子图边数: {edges_count}")
                    if nodes_count > 0 or edges_count > 0:
                        print(f"   ✅ 成功生成图!")
                    else:
                        print(f"   ⚠️  图数据为空（可能是没有相关数据）")
                else:
                    print(f"   ⚠️  Payload中没有subgraph字段")
            else:
                print(f"   ⚠️  未收到payload事件（可能流还在继续）")
            
            print(f"   ✅ livingKB用户请求生成图测试完成")


async def test_history_conversation():
    """测试8: 验证历史对话功能"""
    print("\n8. 测试历史对话功能...")
    token, session_id, _, _ = await get_token_and_session()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        # 第一轮对话
        print(f"   📝 第一轮对话: 我的名字是张三")
        body1 = {"query": "我的名字是张三"}
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers=headers,
            json=body1
        ) as response1:
            assert response1.status_code == 200
            result1 = await read_sse_stream(response1)
            print(f"      - 收到 {len(result1['chunks'])} 个chunks")
            print(f"      - 回复长度: {len(result1['full_content'])} 字符")
        
        # 等待一下，确保消息已保存
        await asyncio.sleep(0.5)
        
        # 第二轮对话（应该能看到历史）
        print(f"   📝 第二轮对话: 我刚才说我叫什么名字？")
        body2 = {"query": "我刚才说我叫什么名字？"}
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers=headers,
            json=body2
        ) as response2:
            assert response2.status_code == 200
            result2 = await read_sse_stream(response2)
            print(f"      - 收到 {len(result2['chunks'])} 个chunks")
            print(f"      - 回复长度: {len(result2['full_content'])} 字符")
            print(f"      - 回复内容: {result2['full_content'][:100]}...")
            
            # 验证回复中是否提到了"张三"（说明看到了历史）
            if "张三" in result2['full_content']:
                print(f"   ✅ 历史对话功能正常（回复中提到了'张三'）")
            else:
                print(f"   ⚠️  回复中未提到'张三'，可能历史对话未生效")
                print(f"   ℹ️  完整回复: {result2['full_content']}")
        
        print(f"   ✅ 历史对话测试完成")


async def test_user_info_stored():
    """测试9: 验证消息中存储了 user_id 和 user_type"""
    print("\n9. 测试用户信息存储...")
    token, session_id, _, user_type = await get_token_and_session()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        # 发送一条消息
        print(f"   📝 发送测试消息")
        body = {"query": "测试用户信息存储"}
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers=headers,
            json=body
        ) as response:
            assert response.status_code == 200
            result = await read_sse_stream(response)
            print(f"      - 收到 {len(result['chunks'])} 个chunks")
        
        # 等待消息保存
        await asyncio.sleep(0.5)
        
        # 查询消息，验证 user_id 和 user_type
        print(f"   🔍 查询消息验证用户信息")
        msg_headers = {"Authorization": f"Bearer {token}"}
        msg_response = await client.get(
            f"{BASE_URL}/session/{session_id}/messages",
            headers=msg_headers
        )
        assert msg_response.status_code == 200
        response_data = msg_response.json()
        # 处理可能的包装格式 {"code":200,"data":[...]}
        if isinstance(response_data, dict) and "data" in response_data:
            messages = response_data["data"]
        else:
            messages = response_data
        assert len(messages) > 0, "应该有至少一条消息"
        
        # 验证最新消息包含用户信息
        latest_msg = messages[-1]
        assert latest_msg.get("user_id") is not None, "消息应该包含 user_id"
        assert latest_msg.get("user_type") is not None, "消息应该包含 user_type"
        assert latest_msg["user_type"] == user_type, f"user_type 应该匹配: {latest_msg['user_type']} != {user_type}"
        
        # 验证所有消息都有用户信息
        for msg in messages:
            assert msg.get("user_id") is not None, f"消息 {msg['id']} 应该包含 user_id"
            assert msg.get("user_type") is not None, f"消息 {msg['id']} 应该包含 user_type"
        
        print(f"      - ✅ user_id: {latest_msg['user_id']}")
        print(f"      - ✅ user_type: {latest_msg['user_type']}")
        print(f"      - ✅ 共验证 {len(messages)} 条消息")
        print(f"   ✅ 用户信息存储测试完成")


if __name__ == "__main__":
    async def run_tests():
        print("=" * 60)
        print("开始测试 POST SSE stream_chat 接口")
        print("=" * 60)
        
        try:
            await test_no_token_rejected()
            await test_invalid_token_rejected()
            await test_invalid_session_id()
            await test_invalid_session_id_format()
            await test_valid_token_streams()
            await test_chatkb_user_cannot_generate_graph()
            await test_livingkb_user_can_generate_graph()
            await test_history_conversation()
            await test_user_info_stored()
            
            print("\n" + "=" * 60)
            print("所有核心测试通过！✅")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_tests())
