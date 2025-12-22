"""
响应包装器中间件：在所有 JSON 响应体最外层添加 request_id 字段
"""
import json
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from asgi_correlation_id import correlation_id


class RequestIdResponseWrapper(BaseHTTPMiddleware):
    """在所有 JSON 响应体最外层添加 request_id 字段"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # 只处理 JSON 响应（检查 content-type）
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            # 获取当前请求的 correlation_id（优先从响应头获取，因为中间件已设置）
            request_id = response.headers.get("X-Request-ID") or correlation_id.get() or "NO-ID"
            
            # 获取原始响应体
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            try:
                # 解析原始 JSON
                original_data = json.loads(body.decode('utf-8'))
                
                # 包装响应：在最外层添加 request_id
                if isinstance(original_data, dict):
                    # 字典：直接展开（保持原有结构）
                    wrapped_data = {
                        "request_id": request_id,
                        **original_data
                    }
                elif isinstance(original_data, list):
                    # 列表：包装成 {"request_id": ..., "data": [...]}
                    wrapped_data = {
                        "request_id": request_id,
                        "data": original_data
                    }
                else:
                    # 其他类型（str, int等）：包装在 data 字段中
                    wrapped_data = {
                        "request_id": request_id,
                        "data": original_data
                    }
                
                # 创建新的 JSON 响应（JSONResponse 会自动计算 Content-Length）
                new_headers = dict(response.headers)
                # 移除旧的 Content-Length，让 JSONResponse 重新计算
                new_headers.pop("content-length", None)
                
                return JSONResponse(
                    content=wrapped_data,
                    status_code=response.status_code,
                    headers=new_headers,
                    media_type=response.media_type
                )
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                # 如果不是有效的 JSON，返回原始响应
                return response
        
        return response
