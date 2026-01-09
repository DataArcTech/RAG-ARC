import os
import logging
from typing import Dict, Any, Optional
import httpx

from fastapi import APIRouter, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


async def _query_openrouter_balance(api_key: str) -> Optional[float]:
    """查询 OpenRouter 账户余额"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/DataArcTech/RAG-ARC",
                }
            )
            response.raise_for_status()
            data = response.json()
            # OpenRouter 返回的数据结构可能包含余额信息
            # 根据实际 API 文档调整字段名
            logger.debug(f"OpenRouter API response: {data}")
            balance = data.get("data", {}).get("balance") or data.get("balance")
            if balance is not None:
                return float(balance)
            # 如果找不到余额，记录完整响应以便调试
            logger.warning(f"OpenRouter balance not found in response: {data}")
            return None
    except Exception as e:
        logger.error(f"Failed to query OpenRouter balance: {e}", exc_info=True)
        return None


async def _query_gptsapi_balance(api_key: str) -> Optional[float]:
    """查询 GPTsAPI 账户余额"""
    try:
        # GPTsAPI 的具体 API 端点需要根据实际文档调整
        # 这里使用假设的端点，需要根据实际情况修改
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.gptsapi.net/v1/balance",
                headers={
                    "Authorization": f"Bearer {api_key}",
                }
            )
            response.raise_for_status()
            data = response.json()
            # 根据实际 API 返回结构调整字段名
            balance = data.get("balance") or data.get("data", {}).get("balance")
            if balance is not None:
                return float(balance)
            return None
    except Exception as e:
        logger.error(f"Failed to query GPTsAPI balance: {e}")
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
        openrouter_balance = await _query_openrouter_balance(openrouter_key)
        result["openrouter"] = {
            "enabled": True,
            "balance": openrouter_balance,
            "status": "success" if openrouter_balance is not None else "failed",
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
        gptsapi_balance = await _query_gptsapi_balance(gptsapi_key)
        result["gptsapi"] = {
            "enabled": True,
            "balance": gptsapi_balance,
            "status": "success" if gptsapi_balance is not None else "failed",
        }
    else:
        result["gptsapi"] = {
            "enabled": False,
            "balance": None,
            "status": "not_configured",
        }
    
    return result

