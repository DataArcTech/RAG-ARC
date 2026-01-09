import os
import logging
from typing import Dict, Any, Optional
import httpx

from fastapi import APIRouter, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


async def _query_openrouter_balance(api_key: str) -> Optional[Dict[str, Any]]:
    """
    查询 OpenRouter 账户使用量和余额信息
    
    OpenRouter 通过 /v1/auth/key 接口返回 API Key 信息
    包含：limit（总额度）、limit_remaining（剩余额度）、usage（已使用量）
    
    Returns:
        Dict containing: used, limit, remaining, raw_data
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/DataArcTech/RAG-ARC",
                    "Content-Type": "application/json",
                }
            )
            response.raise_for_status()
            data = response.json()
            logger.debug(f"OpenRouter API response: {data}")
            
            # 从 data.data 中提取信息（实际响应结构是 {"data": {...}}）
            key_info = data.get("data", {}) if isinstance(data.get("data"), dict) else data
            
            # OpenRouter 返回的字段：limit, limit_remaining, usage
            limit = key_info.get("limit")
            limit_remaining = key_info.get("limit_remaining")
            usage = key_info.get("usage", 0)
            
            # 如果 limit_remaining 存在，直接使用；否则计算 remaining = limit - usage
            if limit_remaining is not None:
                remaining = float(limit_remaining)
            elif limit is not None:
                remaining = float(limit) - float(usage)
            else:
                remaining = None
            
            return {
                "used": float(usage) if usage is not None else 0.0,
                "limit": float(limit) if limit is not None else None,
                "remaining": remaining,
                "limit_remaining": float(limit_remaining) if limit_remaining is not None else None,
                "raw_data": data
            }
    except Exception as e:
        logger.error(f"Failed to query OpenRouter balance: {e}", exc_info=True)
        return None


async def _query_gptsapi_balance(api_key: str) -> Optional[Dict[str, Any]]:
    """
    查询 GPTsAPI 账户余额
    
    GPTsAPI 通过 /v1/user/balance 接口返回余额信息
    直接返回 balance 字段表示当前余额
    
    Returns:
        Dict containing: balance, raw_data
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 主要接口：/v1/user/balance
            url = "https://api.gptsapi.net/v1/user/balance"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"GPTsAPI API response: {data}")
            
            # GPTsAPI 直接返回 balance 字段
            balance = data.get("balance")
            if balance is not None:
                return {
                    "balance": float(balance),
                    "raw_data": data
                }
            
            # 如果找不到 balance，尝试其他可能的字段
            logger.warning(f"GPTsAPI balance not found in response: {data}")
            return None
    except httpx.HTTPStatusError as e:
        # 如果是 404，尝试备用接口
        if e.response.status_code == 404:
            logger.info("GPTsAPI /v1/user/balance returned 404, trying alternative endpoints...")
            return await _query_gptsapi_balance_alt(api_key)
        logger.error(f"Failed to query GPTsAPI balance: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Failed to query GPTsAPI balance: {e}", exc_info=True)
        return None


async def _query_gptsapi_balance_alt(api_key: str) -> Optional[Dict[str, Any]]:
    """GPTsAPI 备用余额查询接口"""
    alt_urls = [
        "https://api.gptsapi.net/v1/account/balance",
        "https://api.gptsapi.net/v1/api-key/info",
        "https://api.gptsapi.net/v1/user/profile"
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    for url in alt_urls:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"GPTsAPI balance found via alternative endpoint: {url}")
                    # 尝试从不同可能的字段中提取余额
                    balance = data.get("balance") or data.get("data", {}).get("balance")
                    if balance is not None:
                        return {
                            "balance": float(balance),
                            "raw_data": data
                        }
        except Exception as e:
            logger.debug(f"GPTsAPI alternative endpoint {url} failed: {e}")
            continue
    
    logger.warning("All GPTsAPI alternative endpoints failed")
    return None


@router.get("/balance", status_code=status.HTTP_200_OK)
async def get_balance():
    """
    查询所有配置的 API 提供商余额
    
    注意：此接口不需要认证，用于系统监控
    
    Returns:
        Dict containing balance information for each provider
    """
    
    result: Dict[str, Any] = {}
    
    # 查询 OpenRouter 余额
    # 根据 CHAT_API_BASE_URL 判断是否使用 OpenRouter
    chat_base_url = os.getenv("CHAT_API_BASE_URL", "")
    openrouter_key = None
    if "openrouter.ai" in chat_base_url:
        openrouter_key = os.getenv("CHAT_API_KEY")
        logger.info(f"OpenRouter detected: base_url={chat_base_url}, key_prefix={openrouter_key[:10] if openrouter_key else None}...")
    
    if openrouter_key:
        openrouter_data = await _query_openrouter_balance(openrouter_key)
        if openrouter_data:
            # 优先使用 limit_remaining，如果不存在则使用 remaining
            balance_value = openrouter_data.get("limit_remaining") or openrouter_data.get("remaining")
            result["openrouter"] = {
                "enabled": True,
                "status": "success",
                "used": openrouter_data.get("used"),
                "limit": openrouter_data.get("limit"),
                "remaining": balance_value,
                "balance": balance_value,  # 兼容字段，使用剩余额度作为余额
            }
        else:
            result["openrouter"] = {
                "enabled": True,
                "status": "failed",
                "balance": None,
            }
    else:
        result["openrouter"] = {
            "enabled": False,
            "balance": None,
            "status": "not_configured",
        }
    
    # 查询 GPTsAPI 余额
    # 根据 EMBEDDING_API_BASE_URL 或 OPENAI_BASE_URL 判断是否使用 GPTsAPI
    embedding_base_url = os.getenv("EMBEDDING_API_BASE_URL", "")
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    gptsapi_key = None
    if "gptsapi.net" in embedding_base_url:
        gptsapi_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        logger.info(f"GPTsAPI detected via EMBEDDING_API_BASE_URL: {embedding_base_url}, key_prefix={gptsapi_key[:10] if gptsapi_key else None}...")
    elif "gptsapi.net" in openai_base_url:
        gptsapi_key = os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_API_KEY")
        logger.info(f"GPTsAPI detected via OPENAI_BASE_URL: {openai_base_url}, key_prefix={gptsapi_key[:10] if gptsapi_key else None}...")
    
    if gptsapi_key:
        gptsapi_data = await _query_gptsapi_balance(gptsapi_key)
        if gptsapi_data:
            result["gptsapi"] = {
                "enabled": True,
                "status": "success",
                "balance": gptsapi_data.get("balance"),
            }
        else:
            result["gptsapi"] = {
                "enabled": True,
                "status": "failed",
                "balance": None,
            }
    else:
        result["gptsapi"] = {
            "enabled": False,
            "balance": None,
            "status": "not_configured",
        }
    
    return result

