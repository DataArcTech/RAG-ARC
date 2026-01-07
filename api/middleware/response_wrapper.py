"""
响应包装器中间件：统一包装所有响应为 StandardResponse 格式
"""
import json
import logging
import time
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from asgi_correlation_id import correlation_id

logger = logging.getLogger(__name__)


class RequestIdResponseWrapper(BaseHTTPMiddleware):
    """在所有 JSON 响应体最外层添加 request_id 字段，并记录请求日志"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取请求信息
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # 确保 correlation_id 上下文中有值（从请求头或 correlation_id 获取）
        # CorrelationIdMiddleware 应该已经设置了，但为了确保日志能正确获取，这里也检查一下
        current_correlation_id = correlation_id.get()
        request_id_from_header = request.headers.get("X-Request-ID")
        
        # 如果 correlation_id 为空，尝试从请求头获取或生成新的
        if not current_correlation_id:
            if request_id_from_header:
                correlation_id.set(request_id_from_header)
            else:
                # 如果都没有，生成一个新的（这种情况不应该发生，但作为兜底）
                new_id = str(uuid.uuid4())
                correlation_id.set(new_id)
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 获取 request_id（优先从响应头获取，因为 CorrelationIdMiddleware 已设置）
        request_id = response.headers.get("X-Request-ID") or correlation_id.get() or "NO-ID"
        
        # 确保 correlation_id 上下文中有值，以便日志记录能正确获取
        # 如果 correlation_id 为空但 request_id 有值，则设置 correlation_id
        if request_id != "NO-ID" and not correlation_id.get():
            correlation_id.set(request_id)
        # 如果 request_id 是 NO-ID 但 correlation_id 有值，使用 correlation_id
        elif request_id == "NO-ID" and correlation_id.get():
            request_id = correlation_id.get()
        
        # 只处理 JSON 响应（检查 content-type）
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
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
                
                # 记录请求日志（包含完整响应内容）
                response_json = json.dumps(wrapped_data, ensure_ascii=False)
                logger.info(
                    f"{client_ip} - \"{method} {path} HTTP/1.1\" {response.status_code} "
                    f"(process_time: {process_time:.3f}s) - Response: {response_json}"
                )
                
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
                # 记录请求日志（非 JSON 响应）
                logger.info(
                    f"{client_ip} - \"{method} {path} HTTP/1.1\" {response.status_code} "
                    f"(process_time: {process_time:.3f}s) - Response: [Non-JSON response]"
                )
                return response
        
        # 记录请求日志（非 JSON 响应）
        logger.info(
            f"{client_ip} - \"{method} {path} HTTP/1.1\" {response.status_code} "
            f"(process_time: {process_time:.3f}s) - Response: [Non-JSON response]"
        )
        return response
