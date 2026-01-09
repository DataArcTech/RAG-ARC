"""
测试告警余额查询接口

测试内容：
1. 测试余额查询接口需要认证
2. 测试余额查询返回格式
3. 测试 OpenRouter 和 GPTsAPI 的余额查询
"""
import os
import httpx
import pytest
from typing import Dict, Any


if os.getenv("RUN_RAGARC_INTEGRATION_TESTS") != "1":
    pytest.skip("Requires a running API server; set RUN_RAGARC_INTEGRATION_TESTS=1 to enable.", allow_module_level=True)


BASE_URL = os.getenv("RAGARC_TEST_BASE_URL", "http://localhost:8001")
TEST_USERNAME = "test_alerts_user"
TEST_PASSWORD = "test_password_123"
TEST_TYPE = 0  # livingKB


async def login_and_get_token() -> str:
    """登录获取token"""
    import time
    username = f"{TEST_USERNAME}_{int(time.time())}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 先注册用户
        register_data = {
            "name": f"测试用户_{int(time.time())}",
            "user_name": username,
            "password": TEST_PASSWORD,
            "type": TEST_TYPE
        }
        try:
            register_response = await client.post(f"{BASE_URL}/auth/register", json=register_data)
            # 注册成功或用户已存在都可以继续
        except Exception:
            pass
        
        # 登录
        login_data = {
            "username": username,
            "password": TEST_PASSWORD
        }
        login_response = await client.post(f"{BASE_URL}/auth/token", data=login_data)
        assert login_response.status_code == 200, f"Login failed: {login_response.text}"
        login_result = login_response.json()
        assert login_result.get("code") == 200, f"Login failed: {login_result}"
        token = login_result["data"]["access_token"]
        return token


@pytest.mark.asyncio
async def test_get_balance_requires_auth():
    """测试余额查询接口需要认证"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{BASE_URL}/api/alerts/balance")
        assert response.status_code == 401, "Should require authentication"


@pytest.mark.asyncio
async def test_get_balance_with_auth():
    """测试余额查询接口（已认证）"""
    token = await login_and_get_token()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(f"{BASE_URL}/api/alerts/balance", headers=headers)
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        result = response.json()
        
        # 验证返回格式
        assert isinstance(result, dict), "Response should be a dict"
        
        # 验证包含 openrouter 和 gptsapi 字段
        assert "openrouter" in result, "Response should contain 'openrouter' field"
        assert "gptsapi" in result, "Response should contain 'gptsapi' field"
        
        # 验证 openrouter 结构
        openrouter = result["openrouter"]
        assert isinstance(openrouter, dict), "openrouter should be a dict"
        assert "enabled" in openrouter, "openrouter should have 'enabled' field"
        assert "status" in openrouter, "openrouter should have 'status' field"
        
        # 验证 gptsapi 结构
        gptsapi = result["gptsapi"]
        assert isinstance(gptsapi, dict), "gptsapi should be a dict"
        assert "enabled" in gptsapi, "gptsapi should have 'enabled' field"
        assert "status" in gptsapi, "gptsapi should have 'status' field"
        
        # 如果配置了 API key，status 应该是 success 或 failed
        # 如果未配置，status 应该是 not_configured
        for provider_name, provider_data in [("openrouter", openrouter), ("gptsapi", gptsapi)]:
            status = provider_data["status"]
            assert status in ["success", "failed", "not_configured"], \
                f"{provider_name} status should be one of: success, failed, not_configured"
            
            if provider_data["enabled"]:
                # 如果 enabled 为 True，balance 字段应该存在（可能为 None）
                assert "balance" in provider_data, f"{provider_name} should have 'balance' field when enabled"
            else:
                # 如果 enabled 为 False，应该是未配置状态
                assert status == "not_configured", \
                    f"{provider_name} should be 'not_configured' when disabled"


@pytest.mark.asyncio
async def test_get_balance_response_structure():
    """测试余额查询返回结构的完整性"""
    token = await login_and_get_token()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(f"{BASE_URL}/api/alerts/balance", headers=headers)
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证返回的是有效的 JSON
        assert result is not None
        
        # 打印结果以便调试
        print(f"\nBalance query result: {result}")

