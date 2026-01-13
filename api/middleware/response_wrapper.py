"""
Response wrapper middleware: normalize JSON responses into StandardResponse shape.
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

_SKIP_JSON_WRAP_PATHS = frozenset({"/openapi.json"})


class RequestIdResponseWrapper(BaseHTTPMiddleware):
    """Add `request_id` to JSON responses and log requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Record request start time.
        start_time = time.time()
        
        # Collect request info for logging.
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        
        # Ensure `correlation_id` is present (from header or existing context).
        # CorrelationIdMiddleware should set this, but we still double-check to keep logs consistent.
        current_correlation_id = correlation_id.get()
        request_id_from_header = request.headers.get("X-Request-ID")
        
        # If missing, try the request header or generate a new one.
        if not current_correlation_id:
            if request_id_from_header:
                correlation_id.set(request_id_from_header)
            else:
                # Should be rare, but still generate a new ID as a safeguard.
                new_id = str(uuid.uuid4())
                correlation_id.set(new_id)
        
        # Resolve request_id (used for downstream error handling).
        request_id = correlation_id.get() or request_id_from_header or str(uuid.uuid4())
        
        # Handle request and catch unexpected errors.
        try:
            response = await call_next(request)
        except Exception as exc:
            # Catch all unhandled exceptions and return a normalized error response.
            process_time = time.time() - start_time
            logger.error(
                f"{client_ip} - \"{method} {path} HTTP/1.1\" 500 "
                f"(process_time: {process_time:.3f}s) - Exception: {exc}",
                exc_info=True
            )
            
            # Return a standard error response.
            error_detail = str(exc) if exc else "Internal server error"
            wrapped_data = {
                "code": 500,
                "message": error_detail,
                "data": None,
                "request_id": request_id
            }
            
            logger.info(
                f"{client_ip} - \"{method} {path} HTTP/1.1\" 500 "
                f"(process_time: {process_time:.3f}s) - Response: code=500 message={error_detail} request_id={request_id}"
            )
            
            return JSONResponse(
                content=wrapped_data,
                status_code=500,
                headers={"X-Request-ID": request_id}
            )
        
        # Compute processing time.
        process_time = time.time() - start_time
        
        # Resolve request_id (prefer response header set by CorrelationIdMiddleware).
        request_id = response.headers.get("X-Request-ID") or correlation_id.get() or "NO-ID"
        
        # Keep correlation_id in sync so log handlers can always read it.
        # If correlation_id is empty but request_id is available, set correlation_id.
        if request_id != "NO-ID" and not correlation_id.get():
            correlation_id.set(request_id)
        # If request_id is missing but correlation_id exists, use correlation_id.
        elif request_id == "NO-ID" and correlation_id.get():
            request_id = correlation_id.get()
        
        # Ensure correlation_id is set before logging (critical to keep request_id observable).
        if request_id != "NO-ID":
            correlation_id.set(request_id)
        
        # Only wrap JSON responses (based on content-type).
        content_type = response.headers.get("content-type", "")
        content_disposition = response.headers.get("content-disposition") or ""
        if path in _SKIP_JSON_WRAP_PATHS:
            logger.info(
                f"{client_ip} - \"{method} {path} HTTP/1.1\" {response.status_code} "
                f"(process_time: {process_time:.3f}s) - Response: [Skip JSON wrapping]"
            )
            return response
        if "application/json" in content_type and not content_disposition:
            # Read the original response body.
            body_chunks: list[bytes] = []
            async for chunk in response.body_iterator:
                body_chunks.append(chunk)
            body = b"".join(body_chunks)
            
            try:
                # Parse original JSON.
                original_data = json.loads(body.decode('utf-8'))
                
                # Normalize into standard response shape.
                # If the payload is already in standard shape (code/message/data), reuse it.
                if isinstance(original_data, dict) and "code" in original_data and "message" in original_data:
                    # Already standard; only add request_id.
                    wrapped_data = {
                        **original_data,
                        "request_id": request_id
                    }
                else:
                    # Wrap into standard format.
                    # For error responses, try to extract message from `data.detail`.
                    message = "success" if response.status_code < 400 else "error"
                    if response.status_code >= 400 and isinstance(original_data, dict):
                        # Prefer `detail` as message.
                        detail = original_data.get("detail") or original_data.get("message")
                        if detail:
                            message = detail
                    
                    wrapped_data = {
                        "code": response.status_code,
                        "message": message,
                        "data": original_data,
                        "request_id": request_id
                    }
                
                # Log request (avoid logging response payload to prevent token leakage / huge bodies).
                data_summary: str
                data_value = wrapped_data.get("data")
                if isinstance(data_value, dict):
                    data_summary = f"dict(keys={sorted(list(data_value.keys()))})"
                elif isinstance(data_value, list):
                    data_summary = f"list(len={len(data_value)})"
                else:
                    data_summary = str(type(data_value).__name__)
                logger.info(
                    f"{client_ip} - \"{method} {path} HTTP/1.1\" {response.status_code} "
                    f"(process_time: {process_time:.3f}s) - Response: code={wrapped_data.get('code')} message={wrapped_data.get('message')} data={data_summary} request_id={request_id}"
                )
                
                # Create a new JSON response (JSONResponse recalculates Content-Length).
                new_headers = dict(response.headers)
                # Remove old Content-Length so JSONResponse can recalculate it.
                new_headers.pop("content-length", None)
                
                return JSONResponse(
                    content=wrapped_data,
                    status_code=response.status_code,
                    headers=new_headers,
                    media_type=response.media_type
                )
            except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                # If the body is not valid JSON (e.g. file download mis-labeled as JSON), return the original payload.
                new_headers = dict(response.headers)
                new_headers.pop("content-length", None)
                logger.info(
                    f"{client_ip} - \"{method} {path} HTTP/1.1\" {response.status_code} "
                    f"(process_time: {process_time:.3f}s) - Response: [Non-JSON response]"
                )
                return Response(
                    content=body,
                    status_code=response.status_code,
                    headers=new_headers,
                    media_type=response.media_type,
                )
        
        # Log request (non-JSON response).
        logger.info(
            f"{client_ip} - \"{method} {path} HTTP/1.1\" {response.status_code} "
            f"(process_time: {process_time:.3f}s) - Response: [Non-JSON response]"
        )
        return response
