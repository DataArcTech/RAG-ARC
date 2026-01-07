"""
测试消息存储与获取功能

测试内容：
1. 发送消息后验证消息是否正确存储到数据库
2. 验证消息中包含 user_id 和 user_type 字段
3. 验证用户只能查询自己的消息
4. 验证按会话查询消息功能
"""
import os

import asyncio
import json
import httpx
import uuid
from typing import Dict, Any

import pytest


if os.getenv("RUN_RAGARC_INTEGRATION_TESTS") != "1":
    pytest.skip("Requires a running API server; set RUN_RAGARC_INTEGRATION_TESTS=1 to enable.", allow_module_level=True)


BASE_URL = os.getenv("RAGARC_TEST_BASE_URL", "http://localhost:8001")
TEST_USERNAME = "test_msg_user"
TEST_PASSWORD = "test_password_123"
TEST_TYPE = 1  # chatKB


async def login_and_get_token(username: str = None, password: str = None, user_type: int = None) -> tuple[str, int]:
    """登录获取token（完全复制test_sse_post_stream_chat.py的实现）"""
    import time
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
                # 注册成功后等待一下，确保数据库同步
                await asyncio.sleep(0.5)
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
        data_obj = data.get("data")
        if data_obj is None:
            raise ValueError(f"登录失败，响应中没有data字段: {data}")
        assert "access_token" in data_obj, f"登录响应中没有access_token: {data}"
        # 获取用户类型
        user_type_from_response = data_obj.get("user", {}).get("type", user_type)
        return data_obj["access_token"], user_type_from_response


async def auto_login_and_get_token(user_type: int = None) -> tuple[str, int]:
    """自动注册并登录获取token（使用唯一用户名）"""
    import time
    user_type = user_type or TEST_TYPE
    username = f"{TEST_USERNAME}_{int(time.time())}"
    return await login_and_get_token(username=username, password=TEST_PASSWORD, user_type=user_type)


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
    """读取 SSE 流并解析"""
    result = {
        "chunks": [],
        "content_parts": [],
        "has_done": False,
    }
    
    if response.status_code != 200:
        return result
    
    try:
        async for line in response.aiter_lines():
            if not line:
                continue
            
            if line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
                if data == "[DONE]":
                    result["has_done"] = True
                    break
                
                try:
                    chunk = json.loads(data)
                    result["chunks"].append(chunk)
                    
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = (choices[0] or {}).get("delta") or {}
                        content = delta.get("content")
                        if content:
                            result["content_parts"].append(content)
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"读取SSE流时出错: {e}")
    
    result["full_content"] = "".join(result["content_parts"])
    return result


async def test_message_storage_and_retrieval():
    """测试1: 消息存储与获取"""
    print("\n" + "=" * 60)
    print("测试消息存储与获取功能")
    print("=" * 60)
    
    # 1. 登录并创建会话
    print("\n1. 登录并创建会话...")
    token, user_type = await auto_login_and_get_token()
    session_id = await create_session(token)
    print(f"   ✅ 登录成功，用户类型: {user_type}")
    print(f"   ✅ 会话ID: {session_id}")
    
    # 2. 发送消息（通过SSE接口）
    print("\n2. 通过SSE接口发送消息...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        test_query = "这是一条测试消息，用于验证存储功能"
        body = {"query": test_query}
        
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
            headers=headers,
            json=body
        ) as response:
            assert response.status_code == 200, f"SSE请求失败: {response.status_code}"
            result = await read_sse_stream(response)
            print(f"   ✅ 收到 {len(result['chunks'])} 个chunks")
            print(f"   ✅ 完整内容长度: {len(result['full_content'])} 字符")
    
    # 等待消息保存到数据库
    await asyncio.sleep(0.5)
    
    # 3. 查询会话消息
    print("\n3. 查询会话消息...")
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(
            f"{BASE_URL}/session/{session_id}/messages",
            headers=headers
        )
        assert response.status_code == 200, f"查询消息失败: {response.status_code}"
        response_data = response.json()
        
        # 处理可能的包装格式
        if isinstance(response_data, dict) and "data" in response_data:
            messages = response_data["data"]
        else:
            messages = response_data
        
        assert len(messages) > 0, "应该有至少一条消息"
        print(f"   ✅ 查询到 {len(messages)} 条消息")
        
        # 4. 验证消息字段
        print("\n4. 验证消息字段...")
        user_message = None
        assistant_message = None
        
        for msg in messages:
            if msg["content"]["role"] == "user":
                user_message = msg
            elif msg["content"]["role"] == "assistant":
                assistant_message = msg
        
        # 验证用户消息
        assert user_message is not None, "应该有一条用户消息"
        assert user_message.get("user_id") is not None, "用户消息应该包含 user_id"
        assert user_message.get("user_type") is not None, "用户消息应该包含 user_type"
        assert user_message["user_type"] == user_type, f"user_type 应该匹配: {user_message['user_type']} != {user_type}"
        assert user_message["content"]["content"] == test_query, "用户消息内容应该匹配"
        print(f"   ✅ 用户消息验证通过")
        print(f"      - user_id: {user_message['user_id']}")
        print(f"      - user_type: {user_message['user_type']}")
        print(f"      - content: {user_message['content']['content'][:50]}...")
        
        # 验证助手消息
        assert assistant_message is not None, "应该有一条助手消息"
        assert assistant_message.get("user_id") is not None, "助手消息应该包含 user_id"
        assert assistant_message.get("user_type") is not None, "助手消息应该包含 user_type"
        assert assistant_message["user_type"] == user_type, f"user_type 应该匹配"
        print(f"   ✅ 助手消息验证通过")
        print(f"      - user_id: {assistant_message['user_id']}")
        print(f"      - user_type: {assistant_message['user_type']}")
        print(f"      - content长度: {len(assistant_message['content']['content'])} 字符")
    
    # 5. 测试查询当前用户的所有消息
    print("\n5. 测试查询当前用户的所有消息...")
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(
            f"{BASE_URL}/session/messages",
            headers=headers,
            params={"limit": 100, "offset": 0}
        )
        assert response.status_code == 200, f"查询用户消息失败: {response.status_code}"
        response_data = response.json()
        
        if isinstance(response_data, dict) and "data" in response_data:
            all_messages = response_data["data"]
        else:
            all_messages = response_data
        
        assert len(all_messages) > 0, "应该至少有一条消息"
        print(f"   ✅ 查询到 {len(all_messages)} 条消息")
        
        # 验证所有消息都属于当前用户
        for msg in all_messages:
            assert msg.get("user_id") is not None, "消息应该包含 user_id"
            assert msg.get("user_type") is not None, "消息应该包含 user_type"
            # 注意：这里不能直接比较 user_id，因为 token 中没有 user_id
            # 但可以验证所有消息都有 user_id 和 user_type
        
        print(f"   ✅ 所有消息都包含 user_id 和 user_type 字段")
    
    print("\n" + "=" * 60)
    print("✅ 消息存储与获取功能测试通过！")
    print("=" * 60)


async def test_user_can_only_query_own_messages():
    """测试2: 验证用户只能查询自己的消息"""
    print("\n" + "=" * 60)
    print("测试用户只能查询自己的消息")
    print("=" * 60)
    
    # 创建两个用户
    print("\n1. 创建两个不同的用户...")
    import time
    username1 = f"{TEST_USERNAME}_user1_{int(time.time())}"
    username2 = f"{TEST_USERNAME}_user2_{int(time.time())}"
    
    token1, type1 = await login_and_get_token(username=username1, password=TEST_PASSWORD, user_type=1)
    session_id1 = await create_session(token1)
    print(f"   ✅ 用户1: type={type1}, session={session_id1}")
    
    token2, type2 = await login_and_get_token(username=username2, password=TEST_PASSWORD, user_type=1)
    session_id2 = await create_session(token2)
    print(f"   ✅ 用户2: type={type2}, session={session_id2}")
    
    # 用户1发送消息
    print("\n2. 用户1发送消息...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {token1}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        body = {"query": "这是用户1的测试消息"}
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id1}",
            headers=headers,
            json=body
        ) as response:
            assert response.status_code == 200
            await read_sse_stream(response)
    
    await asyncio.sleep(0.5)
    
    # 用户2发送消息
    print("\n3. 用户2发送消息...")
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {token2}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        body = {"query": "这是用户2的测试消息"}
        async with client.stream(
            "POST",
            f"{BASE_URL}/rag_inference/stream_chat/{session_id2}",
            headers=headers,
            json=body
        ) as response:
            assert response.status_code == 200
            await read_sse_stream(response)
    
    await asyncio.sleep(0.5)
    
    # 用户1查询自己的消息
    print("\n4. 用户1查询自己的消息...")
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token1}"}
        response = await client.get(
            f"{BASE_URL}/session/messages",
            headers=headers
        )
        assert response.status_code == 200
        response_data = response.json()
        if isinstance(response_data, dict) and "data" in response_data:
            user1_messages = response_data["data"]
        else:
            user1_messages = response_data
        
        # 验证用户1只能看到自己的消息
        for msg in user1_messages:
            assert "这是用户1的测试消息" in msg["content"]["content"] or msg["content"]["role"] == "assistant", \
                "用户1应该只能看到自己的消息"
        print(f"   ✅ 用户1查询到 {len(user1_messages)} 条消息（都是自己的）")
    
    # 用户2查询自己的消息
    print("\n5. 用户2查询自己的消息...")
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token2}"}
        response = await client.get(
            f"{BASE_URL}/session/messages",
            headers=headers
        )
        assert response.status_code == 200
        response_data = response.json()
        if isinstance(response_data, dict) and "data" in response_data:
            user2_messages = response_data["data"]
        else:
            user2_messages = response_data
        
        # 验证用户2只能看到自己的消息
        for msg in user2_messages:
            assert "这是用户2的测试消息" in msg["content"]["content"] or msg["content"]["role"] == "assistant", \
                "用户2应该只能看到自己的消息"
        print(f"   ✅ 用户2查询到 {len(user2_messages)} 条消息（都是自己的）")
    
    print("\n" + "=" * 60)
    print("✅ 用户隔离测试通过！")
    print("=" * 60)


async def test_multi_turn_conversation():
    """测试3: 多轮对话测试（使用指定账号 wangshunchi）"""
    print("\n" + "=" * 60)
    print("测试多轮对话功能（账号: wangshunchi）")
    print("=" * 60)
    
    # 1. 使用指定账号登录
    print("\n1. 使用指定账号登录...")
    token, user_type = await login_and_get_token(
        username="wangshunchi",
        password="wangshunchi",
        user_type=1  # chatKB
    )
    print(f"   ✅ 登录成功，用户类型: {user_type}")
    
    # 2. 创建新会话
    print("\n2. 创建新会话...")
    session_id = await create_session(token)
    print(f"   ✅ 会话ID: {session_id}")
    
    # 3. 进行多轮对话
    queries = [
        "你好，请简单介绍一下你自己",
        "你能做什么？",
        "请告诉我你的主要功能",
    ]
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream"
        }
        
        for i, query in enumerate(queries, 1):
            print(f"\n3.{i} 发送第 {i} 轮消息: {query}")
            
            body = {"query": query}
            async with client.stream(
                "POST",
                f"{BASE_URL}/rag_inference/stream_chat/{session_id}",
                headers=headers,
                json=body
            ) as response:
                assert response.status_code == 200, f"SSE请求失败: {response.status_code}"
                result = await read_sse_stream(response)
                print(f"   ✅ 收到 {len(result['chunks'])} 个chunks")
                print(f"   ✅ 助手回复长度: {len(result['full_content'])} 字符")
                if result['full_content']:
                    print(f"   ✅ 助手回复预览: {result['full_content'][:100]}...")
            
            # 等待消息保存
            await asyncio.sleep(0.5)
    
    # 4. 查询完整的对话历史
    print("\n4. 查询完整的对话历史...")
    await asyncio.sleep(1)  # 确保所有消息都已保存
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(
            f"{BASE_URL}/session/{session_id}/messages",
            headers=headers
        )
        assert response.status_code == 200, f"查询消息失败: {response.status_code}"
        response_data = response.json()
        
        if isinstance(response_data, dict) and "data" in response_data:
            messages = response_data["data"]
        else:
            messages = response_data
        
        print(f"   ✅ 查询到 {len(messages)} 条消息")
        
        # 5. 验证对话完整性
        print("\n5. 验证对话完整性...")
        user_messages = [msg for msg in messages if msg["content"]["role"] == "user"]
        assistant_messages = [msg for msg in messages if msg["content"]["role"] == "assistant"]
        
        print(f"   - 用户消息数: {len(user_messages)}")
        print(f"   - 助手消息数: {len(assistant_messages)}")
        
        # 验证每轮对话都有用户消息和助手回复
        assert len(user_messages) == len(queries), f"应该有 {len(queries)} 条用户消息，实际 {len(user_messages)} 条"
        assert len(assistant_messages) == len(queries), f"应该有 {len(queries)} 条助手消息，实际 {len(assistant_messages)} 条"
        
        # 验证消息顺序（应该是 user, assistant, user, assistant...）
        print("\n6. 验证消息顺序和内容...")
        for i, query in enumerate(queries):
            # 找到对应的用户消息
            user_msg = user_messages[i]
            assert user_msg["content"]["content"] == query, f"第 {i+1} 轮用户消息内容不匹配"
            assert user_msg.get("user_id") is not None, f"第 {i+1} 轮用户消息缺少 user_id"
            assert user_msg.get("user_type") is not None, f"第 {i+1} 轮用户消息缺少 user_type"
            
            # 找到对应的助手消息
            assistant_msg = assistant_messages[i]
            assert len(assistant_msg["content"]["content"]) > 0, f"第 {i+1} 轮助手消息内容为空"
            assert assistant_msg.get("user_id") is not None, f"第 {i+1} 轮助手消息缺少 user_id"
            assert assistant_msg.get("user_type") is not None, f"第 {i+1} 轮助手消息缺少 user_type"
            
            print(f"   ✅ 第 {i+1} 轮对话验证通过")
            print(f"      - 用户: {query[:50]}...")
            print(f"      - 助手: {assistant_msg['content']['content'][:50]}...")
        
        # 打印完整对话历史
        print("\n7. 完整对话历史:")
        print("-" * 60)
        for i, msg in enumerate(messages, 1):
            role = msg["content"]["role"]
            content = msg["content"]["content"]
            created_at = msg.get("created_at", "")
            print(f"{i}. [{role}] ({created_at[:19] if created_at else 'N/A'})")
            if role == "user":
                print(f"   {content}")
            else:
                print(f"   {content[:200]}{'...' if len(content) > 200 else ''}")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("✅ 多轮对话测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    async def run_tests():
        try:
            # 使用指定账号进行多轮对话测试
            await test_multi_turn_conversation()
            
            print("\n" + "=" * 60)
            print("✅ 所有测试通过！")
            print("=" * 60)
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_tests())
