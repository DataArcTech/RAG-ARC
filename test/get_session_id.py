"""
快速获取 session_id 用于 Apifox 测试
"""
import asyncio
import httpx
import time

BASE_URL = "http://localhost:8001"
TEST_USERNAME = "test_sse_user"
TEST_PASSWORD = "test_password_123"
TEST_TYPE = 1  # chatKB


async def login_and_get_token():
    """登录获取token"""
    async with httpx.AsyncClient() as client:
        # 先尝试注册（如果已存在会失败，但不影响）
        register_data = {
            "name": f"测试用户_{int(time.time())}",
            "user_name": TEST_USERNAME,
            "password": TEST_PASSWORD,
            "type": TEST_TYPE
        }
        try:
            await client.post(f"{BASE_URL}/auth/register", json=register_data)
            print(f"✅ 注册用户: {TEST_USERNAME}")
        except Exception:
            print(f"ℹ️  用户已存在: {TEST_USERNAME}")
        
        # 登录
        login_data = {
            "user_name": TEST_USERNAME,
            "password": TEST_PASSWORD,
            "type": TEST_TYPE
        }
        response = await client.post(f"{BASE_URL}/auth/token", json=login_data)
        assert response.status_code == 200, f"登录失败: {response.text}"
        data = response.json()
        assert data.get("code") == 200, f"登录返回code不是200: {data}"
        token = data["data"]["access_token"]
        print(f"✅ 登录成功，获取到 token")
        return token


async def create_session(token: str):
    """创建会话，返回 session_id"""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.post(f"{BASE_URL}/session", headers=headers)
        assert response.status_code == 200, f"创建会话失败: {response.text}"
        data = response.json()
        assert data.get("code") == 200, f"创建会话返回code不是200: {data}"
        session_id = str(data.get("data")).strip().strip('"').strip("'")
        return session_id


async def main():
    print("=" * 60)
    print("获取 Session ID 用于 Apifox 测试")
    print("=" * 60)
    
    # 获取 token
    token = await login_and_get_token()
    print(f"\n📝 Token: {token[:50]}...")
    
    # 创建 session
    session_id = await create_session(token)
    print(f"\n✅ Session ID: {session_id}")
    
    print("\n" + "=" * 60)
    print("Apifox 配置信息：")
    print("=" * 60)
    print(f"\n1. 接口路径: POST /rag_inference/stream_chat/{session_id}")
    print(f"\n2. Headers:")
    print(f"   Authorization: Bearer {token}")
    print(f"   Content-Type: application/json")
    print(f"   Accept: text/event-stream")
    print(f"\n3. Body (JSON):")
    print(f'   {{"query": "你好"}}')
    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

