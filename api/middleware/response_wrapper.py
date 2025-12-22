"""
响应包装器中间件：统一包装所有响应为 StandardResponse 格式
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
                
                # 统一包装为标准响应格式
                # 如果已经是标准格式（有code/message/data），直接使用
                if isinstance(original_data, dict) and "code" in original_data and "message" in original_data:
                    # 已经是标准格式，只添加 request_id
                    wrapped_data = {
                        **original_data,
                        "request_id": request_id
                    }
                else:
                    # 包装为标准格式
                    # 对于错误响应，尝试从 data.detail 提取 message
                    message = "success" if response.status_code < 400 else "error"
                    if response.status_code >= 400 and isinstance(original_data, dict):
                        # 优先使用 detail 字段作为 message
                        detail = original_data.get("detail") or original_data.get("message")
                        if detail:
                            message = detail
                    
                    wrapped_data = {
                        "code": response.status_code,
                        "message": message,
                        "data": original_data,
                        "request_id": request_id
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
