import os
import logging
from typing import Dict, Any, Optional
import httpx

from fastapi import APIRouter, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts", tags=["alerts"])


async def _query_openrouter_balance(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Query OpenRouter account usage and balance.
    
    OpenRouter returns API key information via `/v1/auth/key`, including:
    - limit (total limit)
    - limit_remaining (remaining limit)
    - usage (consumed usage)
    
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
            
            # Extract info from data.data (actual response shape is {"data": {...}}).
            key_info = data.get("data", {}) if isinstance(data.get("data"), dict) else data
            
            # OpenRouter fields: limit, limit_remaining, usage.
            limit = key_info.get("limit")
            limit_remaining = key_info.get("limit_remaining")
            usage = key_info.get("usage", 0)
            
            # Prefer limit_remaining; otherwise compute remaining = limit - usage.
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


@router.get("/balance", status_code=status.HTTP_200_OK)
async def get_balance():
    """
    Query OpenRouter account balance.
    
    Note: this endpoint is unauthenticated and intended for system monitoring.
    
    Returns:
        Dict containing OpenRouter balance information
    """
    
    result: Dict[str, Any] = {}
    
    # Query OpenRouter balance.
    # Determine whether OpenRouter is in use via CHAT_API_BASE_URL.
    chat_base_url = os.getenv("CHAT_API_BASE_URL", "")
    openrouter_key = None
    if "openrouter.ai" in chat_base_url:
        openrouter_key = os.getenv("CHAT_API_KEY")
        logger.info(f"OpenRouter detected: base_url={chat_base_url}, key_prefix={openrouter_key[:10] if openrouter_key else None}...")
    
    if openrouter_key:
        openrouter_data = await _query_openrouter_balance(openrouter_key)
        if openrouter_data:
            # Prefer limit_remaining; if missing, fall back to remaining.
            balance_value = openrouter_data.get("limit_remaining") or openrouter_data.get("remaining")
            result["openrouter"] = {
                "enabled": True,
                "status": "success",
                "used": openrouter_data.get("used"),
                "limit": openrouter_data.get("limit"),
                "remaining": balance_value,
                "balance": balance_value,  # Compatibility field: use remaining limit as balance.
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
    
    return result
