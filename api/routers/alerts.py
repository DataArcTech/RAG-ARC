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
    
    GPTsAPI 通过 /user/balanceInfo 接口返回余额信息
    使用 api2.gptsapi.net 域名
    
    Returns:
        Dict containing: balance, total_recharge, consumed, raw_data
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 正确的接口：/user/balanceInfo，使用 api2.gptsapi.net
            url = "https://api2.gptsapi.net/user/balanceInfo"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"GPTsAPI API response: {data}")
            
            # GPTsAPI 返回的字段可能是 balance 或其他字段名
            # 根据实际响应结构调整
            balance = data.get("balance") or data.get("data", {}).get("balance")
            total_recharge = data.get("total_recharge") or data.get("data", {}).get("total_recharge")
            consumed = data.get("consumed") or data.get("data", {}).get("consumed")
            
            if balance is not None:
                return {
                    "balance": float(balance),
                    "total_recharge": float(total_recharge) if total_recharge is not None else None,
                    "consumed": float(consumed) if consumed is not None else None,
                    "raw_data": data
                }
            
            # 如果找不到 balance，记录完整响应以便调试
            logger.warning(f"GPTsAPI balance not found in response: {data}")
            return None
    except Exception as e:
        logger.error(f"Failed to query GPTsAPI balance: {e}", exc_info=True)
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

