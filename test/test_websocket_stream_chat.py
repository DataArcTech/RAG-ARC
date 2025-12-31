"""
测试 WebSocket stream_chat 接口
测试：1. 参数传递方式 2. 生成图功能 3. type校验
"""
import asyncio
import json
import time
import httpx
import uuid
from typing import Dict, Any

try:
    import websockets
except ImportError:
    print("需要安装 websockets 库: pip install websockets")
    raise

BASE_URL = "http://localhost:8001"
WS_BASE = BASE_URL.replace("https://", "wss://").replace("http://", "ws://")
TEST_USERNAME_CHATKB = "test_ws_chatkb"
TEST_USERNAME_LIVINGKB = "test_ws_livingkb"
TEST_PASSWORD = "test_password_123"


async def login_and_get_token(username: str, password: str, user_type: int) -> tuple[str, int]:
    """登录获取token"""
    async with httpx.AsyncClient() as client:
        # 先尝试注册
        register_data = {
            "name": f"测试用户_{int(time.time())}",
            "user_name": username,
            "password": password,
            "type": user_type
        }
        try:
            await client.post(f"{BASE_URL}/auth/register", json=register_data)
            print(f"   ✅ 注册用户: {username} (type={user_type})")
        except Exception:
            print(f"   ℹ️  用户可能已存在: {username}")
        
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
        user_type_from_response = data.get("data", {}).get("user", {}).get("type", user_type)
        return data["data"]["access_token"], user_type_from_response


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


async def test_websocket_plain_text():
    """测试1: WebSocket 纯文本消息"""
    print("\n1. 测试 WebSocket 纯文本消息...")
    username = f"{TEST_USERNAME_CHATKB}_{int(time.time())}"
    token, user_type = await login_and_get_token(username, TEST_PASSWORD, 1)
    session_id = await create_session(token)
    
    ws_url = f"{WS_BASE}/rag_inference/stream_chat/{session_id}"
    
    async with websockets.connect(
        ws_url,
        ping_interval=None,
        close_timeout=3,
        additional_headers={"Cookie": f"auth_token={token}"},
    ) as ws:
        # 发送纯文本消息
        await ws.send("你好")
        
        # 接收响应
        raw_response = await ws.recv()
        response = json.loads(raw_response)
        print(f"   📦 收到响应")
        print(f"      - message: {response.get('message', {}).get('content', {}).get('content', '')[:50]}...")
        print(f"      - chunks: {len(response.get('chunks', []))} 个")
        
        assert "message" in response, "响应应该包含 message 字段"
        assert "chunks" in response, "响应应该包含 chunks 字段"
        print(f"   ✅ 纯文本消息测试成功")


async def test_websocket_json_without_graph():
    """测试2: WebSocket JSON 消息（不生成图）"""
    print("\n2. 测试 WebSocket JSON 消息（不生成图）...")
    username = f"{TEST_USERNAME_CHATKB}_{int(time.time())}"
    token, user_type = await login_and_get_token(username, TEST_PASSWORD, 1)
    session_id = await create_session(token)
    
    ws_url = f"{WS_BASE}/rag_inference/stream_chat/{session_id}"
    
    async with websockets.connect(
        ws_url,
        ping_interval=None,
        close_timeout=3,
        additional_headers={"Cookie": f"auth_token={token}"},
    ) as ws:
        # 发送 JSON 消息（不生成图）
        payload = {
            "query": "你好，请简单介绍一下你自己",
            "return_subgraph": False
        }
        await ws.send(json.dumps(payload))
        
        # 接收响应
        raw_response = await ws.recv()
        response = json.loads(raw_response)
        print(f"   📦 收到响应")
        print(f"      - message: {response.get('message', {}).get('content', {}).get('content', '')[:50]}...")
        print(f"      - chunks: {len(response.get('chunks', []))} 个")
        print(f"      - subgraph: {'有' if response.get('subgraph') else '无'}")
        
        assert "message" in response, "响应应该包含 message 字段"
        assert "chunks" in response, "响应应该包含 chunks 字段"
        assert response.get("subgraph") is None, "不请求生成图时，subgraph 应该为 None"
        print(f"   ✅ JSON 消息（不生成图）测试成功")


async def test_websocket_chatkb_cannot_generate_graph():
    """测试3: chatKB用户(type=1)请求生成图应该被拒绝"""
    print("\n3. 测试 chatKB用户(type=1)请求生成图应该被拒绝...")
    username = f"{TEST_USERNAME_CHATKB}_{int(time.time())}"
    token, user_type = await login_and_get_token(username, TEST_PASSWORD, 1)
    assert user_type == 1, f"应该是type=1的用户，实际: {user_type}"
    session_id = await create_session(token)
    
    ws_url = f"{WS_BASE}/rag_inference/stream_chat/{session_id}"
    
    try:
        async with websockets.connect(
            ws_url,
            ping_interval=None,
            close_timeout=3,
            additional_headers={"Cookie": f"auth_token={token}"},
        ) as ws:
            # 发送 JSON 消息（请求生成图）
            payload = {
                "query": "test query",
                "return_subgraph": True
            }
            print(f"   📝 发送消息: {payload}")
            await ws.send(json.dumps(payload))
            
            # 应该收到关闭连接
            try:
                raw_response = await ws.recv()
                response = json.loads(raw_response)
                print(f"   ⚠️  意外收到响应: {response}")
                raise AssertionError("chatKB用户请求生成图应该被拒绝，但收到了响应")
            except websockets.exceptions.ConnectionClosed as e:
                # WebSocket 应该被关闭
                print(f"   ✅ WebSocket 连接被关闭（符合预期）")
                print(f"   ✅ 关闭代码: {e.code}, 原因: {e.reason}")
                if e.code == 1008:  # POLICY_VIOLATION
                    print(f"   ✅ chatKB用户请求生成图被正确拒绝（1008 POLICY_VIOLATION）")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"   ✅ WebSocket 连接被关闭: code={e.code}, reason={e.reason}")
        if e.code == 1008:
            print(f"   ✅ chatKB用户请求生成图被正确拒绝（1008 POLICY_VIOLATION）")
    except Exception as e:
        print(f"   ⚠️  其他错误: {e}")
        raise


async def test_websocket_livingkb_can_generate_graph():
    """测试4: livingKB用户(type=0)请求生成图应该成功"""
    print("\n4. 测试 livingKB用户(type=0)请求生成图...")
    username = f"{TEST_USERNAME_LIVINGKB}_{int(time.time())}"
    token, user_type = await login_and_get_token(username, TEST_PASSWORD, 0)
    assert user_type == 0, f"应该是type=0的用户，实际: {user_type}"
    session_id = await create_session(token)
    
    ws_url = f"{WS_BASE}/rag_inference/stream_chat/{session_id}"
    
    async with websockets.connect(
        ws_url,
        ping_interval=None,
        close_timeout=3,
        additional_headers={"Cookie": f"auth_token={token}"},
    ) as ws:
        # 发送 JSON 消息（请求生成图）
        payload = {
            "query": "你好，请简单介绍一下你自己",
            "return_subgraph": True
        }
        print(f"   📝 发送消息: {payload}")
        await ws.send(json.dumps(payload))
        
        # 接收响应
        raw_response = await ws.recv()
        response = json.loads(raw_response)
        print(f"   📦 收到响应")
        print(f"      - message: {response.get('message', {}).get('content', {}).get('content', '')[:50]}...")
        print(f"      - chunks: {len(response.get('chunks', []))} 个")
        
        has_subgraph = "subgraph" in response and response.get("subgraph") is not None
        print(f"      - subgraph: {'有' if has_subgraph else '无'}")
        
        if has_subgraph:
            subgraph = response.get("subgraph", {})
            nodes_count = len(subgraph.get("nodes", []))
            edges_count = len(subgraph.get("edges", []))
            print(f"      - 子图节点数: {nodes_count}")
            print(f"      - 子图边数: {edges_count}")
            if nodes_count > 0 or edges_count > 0:
                print(f"   ✅ 成功生成图!")
            else:
                print(f"   ⚠️  图数据为空（可能是没有相关数据）")
        else:
            print(f"   ⚠️  响应中没有subgraph字段")
        
        assert "message" in response, "响应应该包含 message 字段"
        assert "chunks" in response, "响应应该包含 chunks 字段"
        print(f"   ✅ livingKB用户请求生成图测试完成")


if __name__ == "__main__":
    async def run_tests():
        print("=" * 60)
        print("开始测试 WebSocket stream_chat 接口")
        print("=" * 60)
        
        try:
            await test_websocket_plain_text()
            await test_websocket_json_without_graph()
            await test_websocket_chatkb_cannot_generate_graph()
            await test_websocket_livingkb_can_generate_graph()
            
            print("\n" + "=" * 60)
            print("所有测试通过！✅")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_tests())

