"""
测试认证功能（包含type字段）
测试注册、登录和获取当前用户信息
"""
import asyncio
import time
import httpx
from typing import Dict, Any


BASE_URL = "http://localhost:8000"
current_time = int(time.time())


async def test_register_user_livingkb():
    """测试注册livingKB用户（type=0）"""
    async with httpx.AsyncClient() as client:
        # 注册用户
        register_data = {
            "user_name": f"test_livingkb_{current_time}",
            "password": "test123456",
            "type": 0  # livingKB
        }
        response = await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        assert response.status_code == 201, f"注册失败: {response.text}"
        data = response.json()
        assert data["code"] == 201
        assert data["data"]["user_name"] == register_data["user_name"]
        assert data["data"]["type"] == 0
        assert data["data"]["status"] in ["active", "ACTIVE"]  # 兼容不同格式
        
        return data["data"]


async def test_register_user_chatkb():
    """测试注册chatKB用户（type=1）"""
    async with httpx.AsyncClient() as client:
        # 注册用户
        register_data = {
            "user_name": f"test_chatkb_{current_time}",
            "password": "test123456",
            "type": 1  # chatKB
        }
        response = await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        assert response.status_code == 201, f"注册失败: {response.text}"
        data = response.json()
        assert data["code"] == 201
        assert data["data"]["user_name"] == register_data["user_name"]
        assert data["data"]["type"] == 1
        assert data["data"]["status"] in ["active", "ACTIVE"]  # 兼容不同格式
        
        return data["data"]


async def test_login_livingkb():
    """测试livingKB用户登录（type=0）"""
    async with httpx.AsyncClient() as client:
        # 先注册
        register_data = {
            "user_name": f"login_livingkb_{current_time}",
            "password": "test123456",
            "type": 0
        }
        await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        # 登录（使用表单数据格式）
        login_data = {
            "username": register_data["user_name"],
            "password": register_data["password"],
            "grant_type": "password",
            "type": 0
        }
        response = await client.post(f"{BASE_URL}/auth/token", data=login_data)
        
        assert response.status_code == 200, f"登录失败: {response.text}"
        data = response.json()
        assert data["code"] == 200
        assert "access_token" in data["data"]
        assert data["data"]["token_type"] == "bearer"
        assert data["data"]["type"] == 0
        
        return data["data"]["access_token"]


async def test_login_chatkb():
    """测试chatKB用户登录（type=1）"""
    async with httpx.AsyncClient() as client:
        # 先注册
        register_data = {
            "user_name": f"login_chatkb_{current_time}",
            "password": "test123456",
            "type": 1
        }
        await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        # 登录（使用表单数据格式）
        login_data = {
            "username": register_data["user_name"],
            "password": register_data["password"],
            "grant_type": "password",
            "type": 1
        }
        response = await client.post(f"{BASE_URL}/auth/token", data=login_data)
        
        assert response.status_code == 200, f"登录失败: {response.text}"
        data = response.json()
        assert data["code"] == 200
        assert "access_token" in data["data"]
        assert data["data"]["token_type"] == "bearer"
        assert data["data"]["type"] == 1
        
        return data["data"]["access_token"]


async def test_login_type_mismatch():
    """测试type不匹配的情况"""
    async with httpx.AsyncClient() as client:
        # 先注册type=0的用户
        register_data = {
            "user_name": f"type_test_{current_time}",
            "password": "test123456",
            "type": 0
        }
        await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        # 尝试用type=1登录
        login_data = {
            "username": register_data["user_name"],
            "password": register_data["password"],
            "grant_type": "password",
            "type": 1  # 错误的type
        }
        response = await client.post(f"{BASE_URL}/auth/token", data=login_data)
        
        assert response.status_code == 400, f"应该返回400错误: {response.text}"
        data = response.json()
        assert data["code"] == 400
        assert "type mismatch" in data["message"].lower() or "mismatch" in data["message"].lower()


async def test_get_current_user():
    """测试获取当前用户信息"""
    async with httpx.AsyncClient() as client:
        # 先注册并登录
        register_data = {
            "user_name": f"current_user_{current_time}",
            "password": "test123456",
            "type": 0
        }
        await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        login_data = {
            "username": register_data["user_name"],
            "password": register_data["password"],
            "grant_type": "password",
            "type": 0
        }
        login_response = await client.post(f"{BASE_URL}/auth/token", data=login_data)
        token = login_response.json()["data"]["access_token"]
        
        # 获取当前用户
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(f"{BASE_URL}/user/me", headers=headers)
        
        assert response.status_code == 200, f"获取用户信息失败: {response.text}"
        data = response.json()
        assert data["code"] == 200
        assert data["data"]["user_name"] == register_data["user_name"]
        assert data["data"]["type"] == 0
        assert data["data"]["status"] in ["active", "ACTIVE"]  # 兼容不同格式


async def test_register_without_type():
    """测试注册时不传递type，应该默认为0"""
    async with httpx.AsyncClient() as client:
        # 注册用户，不传递type字段
        register_data = {
            "user_name": f"test_default_type_{current_time}",
            "password": "test123456"
            # 不包含 type 字段
        }
        response = await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        assert response.status_code == 201, f"注册失败: {response.text}"
        data = response.json()
        assert data["code"] == 201
        assert data["data"]["user_name"] == register_data["user_name"]
        assert data["data"]["type"] == 0, "type应该默认为0"
        assert data["data"]["status"] in ["active", "ACTIVE"]  # 兼容不同格式
        
        return data["data"]


async def test_login_without_type():
    """测试登录时不传递type，应该默认为0"""
    async with httpx.AsyncClient() as client:
        # 先注册type=0的用户
        register_data = {
            "user_name": f"login_default_type_{current_time}",
            "password": "test123456",
            "type": 0
        }
        await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        # 登录时不传递type字段
        login_data = {
            "username": register_data["user_name"],
            "password": register_data["password"],
            "grant_type": "password"
            # 不包含 type 字段，应该默认为0
        }
        response = await client.post(f"{BASE_URL}/auth/token", data=login_data)
        
        assert response.status_code == 200, f"登录失败: {response.text}"
        data = response.json()
        assert data["code"] == 200
        assert "access_token" in data["data"]
        assert data["data"]["token_type"] == "bearer"
        assert data["data"]["type"] == 0, "type应该默认为0"
        
        return data["data"]["access_token"]


async def test_login_without_type_mismatch():
    """测试登录时不传递type（默认为0），但用户实际type=1，应该失败"""
    async with httpx.AsyncClient() as client:
        # 先注册type=1的用户
        register_data = {
            "user_name": f"mismatch_test_{current_time}",
            "password": "test123456",
            "type": 1  # chatKB
        }
        await client.post(f"{BASE_URL}/auth/register", json=register_data)
        
        # 登录时不传递type（默认为0），但用户实际是type=1
        login_data = {
            "username": register_data["user_name"],
            "password": register_data["password"],
            "grant_type": "password"
            # 不包含 type 字段，默认为0，但用户是1，应该失败
        }
        response = await client.post(f"{BASE_URL}/auth/token", data=login_data)
        
        assert response.status_code == 400, f"应该返回400错误: {response.text}"
        data = response.json()
        assert data["code"] == 400
        assert "type mismatch" in data["message"].lower() or "mismatch" in data["message"].lower()


if __name__ == "__main__":
    async def run_tests():
        print("=" * 60)
        print("开始测试认证功能（包含type字段）")
        print("=" * 60)
        
        try:
            print("\n1. 测试注册livingKB用户...")
            await test_register_user_livingkb()
            print("   ✅ 通过")
            
            print("\n2. 测试注册chatKB用户...")
            await test_register_user_chatkb()
            print("   ✅ 通过")
            
            print("\n3. 测试livingKB用户登录...")
            await test_login_livingkb()
            print("   ✅ 通过")
            
            print("\n4. 测试chatKB用户登录...")
            await test_login_chatkb()
            print("   ✅ 通过")
            
            print("\n5. 测试type不匹配...")
            await test_login_type_mismatch()
            print("   ✅ 通过")
            
            print("\n6. 测试获取当前用户信息...")
            await test_get_current_user()
            print("   ✅ 通过")
            
            print("\n7. 测试注册时不传递type（应该默认为0）...")
            await test_register_without_type()
            print("   ✅ 通过")
            
            print("\n8. 测试登录时不传递type（应该默认为0）...")
            await test_login_without_type()
            print("   ✅ 通过")
            
            print("\n9. 测试登录时不传递type但用户type不匹配...")
            await test_login_without_type_mismatch()
            print("   ✅ 通过")
            
            print("\n" + "=" * 60)
            print("所有测试通过！✅")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(run_tests())

