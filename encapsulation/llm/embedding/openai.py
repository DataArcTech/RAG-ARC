from .base import EmbeddingLLMBase
from typing import Union, List, Dict, Any, Optional
from encapsulation.llm.utils.openai_client import create_openai_sync_client, create_openai_async_client
from encapsulation.llm.utils.huggingface_client import create_sentence_transformer_client
import logging
import random
import re
import time

logger = logging.getLogger(__name__)


class OpenAIEmbeddingLLM(EmbeddingLLMBase):
    """
    OpenAI Embedding LLM implementation for high-quality text vectorization.

    This class provides a complete embedding solution using OpenAI's API,
    supporting various embedding models with configurable dimensions and parameters.
    Optimized for production use with automatic retry logic and batch processing.

    Key features:
    - Multiple embedding model support: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
    - Configurable embedding dimensions for supported models
    - Batch processing for improved efficiency
    - Automatic retry logic and timeout handling
    - Flexible API endpoint configuration (OpenAI or compatible)

    Configuration options:
        - api_key: OpenAI API key or compatible service key
        - base_url: API endpoint (defaults to OpenAI, supports custom servers)
        - model_name: Target embedding model identifier
        - embedding_dimensions: Custom embedding size (if supported)
        - organization: OpenAI organization ID
        - timeout: Request timeout in seconds
        - max_retries: Automatic retry attempts
    """

    def __init__(self, config):
        """Initialize OpenAI Embedding with loading method support"""
        super().__init__(config)
        # Cache config values to avoid repeated getattr calls
        self.model_name = getattr(self.config, 'model_name', 'text-embedding-ada-002')
        self.embedding_dimensions = getattr(self.config, 'embedding_dimensions', None)
        self.loading_method = getattr(self.config, 'loading_method', 'openai')

        # Initialize client based on loading method
        if self.loading_method == 'openai':
            self.client = create_openai_sync_client(self.config)
            self.async_client = create_openai_async_client(self.config)
        elif self.loading_method == 'huggingface':
            # For HuggingFace sentence transformers
            self.client = create_sentence_transformer_client(self.config)
            self.async_client = None  # HuggingFace uses asyncio.to_thread wrapper
        else:
            raise ValueError(f"Unsupported loading method: {self.loading_method}")

    # ==================== EMBEDDING IMPLEMENTATION ====================

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using OpenAI
        """
        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts or [])

        # Validate inputs
        for idx, text in enumerate(text_list):
            if not self._validate_input(text):
                text_preview_len = len(text) if isinstance(text, str) else None
                raise ValueError(
                    "Text validation failed "
                    f"(index={idx}, type={type(text).__name__}, len={text_preview_len})"
                )

        if self.loading_method == "huggingface":
            try:
                cleaned_texts = [text.replace("\n", " ") for text in text_list]
                encode_kwargs = getattr(self.config, "encode_kwargs", {}) or {}
                embeddings = self.client.encode(cleaned_texts, **encode_kwargs)
                embed_list = embeddings.tolist()
                if is_single:
                    row = embed_list[0] if embed_list else []
                    return [float(x) for x in row]
                return [[float(x) for x in row] for row in embed_list]
            except Exception as e:
                logger.error(f"Embedding failed: {str(e)}")
                raise RuntimeError(f"Embedding failed: {str(e)}")

        def _extract_embeddings(response) -> list[list[float]]:
            if hasattr(response, 'data') and response.data:
                return [item.embedding for item in response.data]
            if isinstance(response, dict) and 'data' in response:
                return [item['embedding'] for item in response['data']]
            raise RuntimeError(f"Unexpected response format: {type(response)}")

        def _looks_like_batch_input_rejected(exc: Exception) -> bool:
            msg = str(exc).lower()
            return (
                "$.input" in msg
                or "input' is invalid" in msg
                or "input is invalid" in msg
                or "invalid input" in msg
            )

        def _embed_with_mode(cleaned_texts: list[str], *, supports_batch_input: bool) -> list[list[float]]:
            embedding_kwargs = {}
            if self.embedding_dimensions:
                embedding_kwargs['dimensions'] = self.embedding_dimensions

            batch_size = int(getattr(self.config, "request_batch_size", 64) or 64)
            batch_size = max(1, batch_size)

            embeddings: list[list[float]] = []

            rate_limit_max_retries = int(getattr(self.config, "rate_limit_max_retries", 6) or 6)
            rate_limit_default_sleep = float(getattr(self.config, "rate_limit_default_sleep_seconds", 60.0) or 60.0)
            rate_limit_max_sleep = float(getattr(self.config, "rate_limit_max_sleep_seconds", 60.0) or 60.0)

            def _retry_after_seconds(exc: Exception) -> float | None:
                def _coerce_float(value: Any) -> float | None:
                    if value is None:
                        return None
                    if isinstance(value, (int, float)):
                        return float(value)
                    text = str(value).strip()
                    if not text:
                        return None
                    try:
                        return float(text)
                    except ValueError:
                        return None

                def _status_code(value: Any) -> int | None:
                    if value is None:
                        return None
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None

                # Prefer structured detection (OpenAI SDK exposes status/headers on APIStatusError subclasses).
                status = _status_code(getattr(exc, "status_code", None))
                if status is None:
                    response = getattr(exc, "response", None)
                    status = _status_code(getattr(response, "status_code", None)) or _status_code(getattr(response, "status", None))

                msg = str(exc)
                is_rate_limited = (status == 429) or ("rate limit" in msg.lower()) or ("429" in msg)
                if not is_rate_limited:
                    return None

                # 1) SDK-provided retry_after (if present)
                retry_after = _coerce_float(getattr(exc, "retry_after", None))
                if retry_after is not None:
                    return max(0.0, retry_after)

                # 2) Response headers: Retry-After / ratelimit reset headers.
                headers = None
                response = getattr(exc, "response", None)
                if response is not None:
                    headers = getattr(response, "headers", None)
                if headers is None:
                    headers = getattr(exc, "headers", None)
                if headers:
                    retry_after_raw = None
                    try:
                        retry_after_raw = headers.get("retry-after") or headers.get("Retry-After")
                    except Exception:
                        retry_after_raw = None
                    retry_after = _coerce_float(retry_after_raw)
                    if retry_after is not None:
                        return max(0.0, retry_after)

                    # OpenAI-compatible gateways sometimes expose "x-ratelimit-reset-requests: 1s".
                    reset_raw = None
                    try:
                        reset_raw = headers.get("x-ratelimit-reset-requests") or headers.get("X-RateLimit-Reset-Requests")
                    except Exception:
                        reset_raw = None
                    if reset_raw:
                        reset_text = str(reset_raw).strip().lower()
                        match = re.match(r"^(\\d+(?:\\.\\d+)?)(ms|s)$", reset_text)
                        if match:
                            amount = _coerce_float(match.group(1))
                            if amount is not None:
                                unit = match.group(2)
                                return max(0.0, amount / 1000.0) if unit == "ms" else max(0.0, amount)

                # 3) Fallback: message parsing (best-effort)
                match = re.search(r"retry after\\s+(\\d+(?:\\.\\d+)?)\\s*seconds", msg, flags=re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None
                return None

            def _call_with_rate_limit_backoff(*, model: str, input_payload: Any) -> Any:
                attempt = 0
                while True:
                    try:
                        return self.client.embeddings.create(model=model, input=input_payload, **embedding_kwargs)
                    except Exception as exc:  # noqa: BLE001
                        retry_after = _retry_after_seconds(exc)
                        if retry_after is None or attempt >= rate_limit_max_retries:
                            raise
                        sleep_for = max(retry_after, rate_limit_default_sleep) if retry_after else rate_limit_default_sleep
                        sleep_for = min(rate_limit_max_sleep, max(0.0, sleep_for) + random.uniform(0.0, 1.0))
                        logger.warning("Embedding rate-limited (429); sleeping %.2fs before retry %d/%d", sleep_for, attempt + 1, rate_limit_max_retries)
                        time.sleep(sleep_for)
                        attempt += 1

            if supports_batch_input:
                for i in range(0, len(cleaned_texts), batch_size):
                    batch = cleaned_texts[i:i + batch_size]
                    response = _call_with_rate_limit_backoff(model=self.model_name, input_payload=batch)
                    embeddings.extend(_extract_embeddings(response))
            else:
                for text in cleaned_texts:
                    response = _call_with_rate_limit_backoff(model=self.model_name, input_payload=text)
                    embeddings.extend(_extract_embeddings(response))
            return embeddings

        try:
            cleaned_texts = [text.replace("\n", " ") for text in text_list]
            supports_batch_input = bool(getattr(self.config, "supports_batch_input", True))
            embeddings = _embed_with_mode(cleaned_texts, supports_batch_input=supports_batch_input)

            # Return single embedding or list based on input
            return embeddings[0] if is_single else embeddings

        except Exception as e:
            # Some OpenAI-compatible endpoints reject list inputs for `input`.
            # Detect and provide an explainable downgrade path.
            supports_batch_input = bool(getattr(self.config, "supports_batch_input", True))
            if (
                supports_batch_input
                and not is_single
                and len(text_list) > 1
                and _looks_like_batch_input_rejected(e)
            ):
                logger.warning(
                    "Embeddings endpoint rejected batched list input; retrying per-item. "
                    "Set EMBEDDING_SUPPORTS_BATCH_INPUT=0 to skip batching."
                )
                try:
                    cleaned_texts = [text.replace("\n", " ") for text in text_list]
                    embeddings = _embed_with_mode(cleaned_texts, supports_batch_input=False)
                    return embeddings[0] if is_single else embeddings
                except Exception as e2:
                    logger.error(f"Embedding failed after per-item retry: {str(e2)}")
                    raise RuntimeError(f"Embedding failed: {str(e2)}") from e2

            logger.error(f"Embedding failed: {str(e)}")
            raise RuntimeError(f"Embedding failed: {str(e)}") from e

    async def aembed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Async function of generate embeddings
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        for text in text_list:
            if not self._validate_input(text):
                raise ValueError(f"Invalid text: {text}")

        if self.loading_method == "huggingface":
            import asyncio

            def _encode_sync():
                cleaned = [text.replace("\n", " ") for text in text_list]
                encode_kwargs = getattr(self.config, "encode_kwargs", {}) or {}
                return self.client.encode(cleaned, **encode_kwargs)

            try:
                embeddings = await asyncio.to_thread(_encode_sync)
                embed_list = embeddings.tolist()
                if is_single:
                    row = embed_list[0] if embed_list else []
                    return [float(x) for x in row]
                return [[float(x) for x in row] for row in embed_list]
            except Exception as e:
                logger.error(f"Embedding failed: {str(e)}")
                raise RuntimeError(f"Embedding failed: {str(e)}")

        try:
            import asyncio

            # Create embedding request
            embedding_kwargs = {}
            if self.embedding_dimensions:
                embedding_kwargs['dimensions'] = self.embedding_dimensions

            embeddings = []
            supports_batch_input = bool(getattr(self.config, "supports_batch_input", True))
            batch_size = int(getattr(self.config, "request_batch_size", 64) or 64)
            batch_size = max(1, batch_size)

            rate_limit_max_retries = int(getattr(self.config, "rate_limit_max_retries", 6) or 6)
            rate_limit_default_sleep = float(getattr(self.config, "rate_limit_default_sleep_seconds", 60.0) or 60.0)
            rate_limit_max_sleep = float(getattr(self.config, "rate_limit_max_sleep_seconds", 60.0) or 60.0)

            def _retry_after_seconds(exc: Exception) -> float | None:
                def _coerce_float(value: Any) -> float | None:
                    if value is None:
                        return None
                    if isinstance(value, (int, float)):
                        return float(value)
                    text = str(value).strip()
                    if not text:
                        return None
                    try:
                        return float(text)
                    except ValueError:
                        return None

                def _status_code(value: Any) -> int | None:
                    if value is None:
                        return None
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return None

                status = _status_code(getattr(exc, "status_code", None))
                if status is None:
                    response = getattr(exc, "response", None)
                    status = _status_code(getattr(response, "status_code", None)) or _status_code(getattr(response, "status", None))

                msg = str(exc)
                is_rate_limited = (status == 429) or ("rate limit" in msg.lower()) or ("429" in msg)
                if not is_rate_limited:
                    return None

                retry_after = _coerce_float(getattr(exc, "retry_after", None))
                if retry_after is not None:
                    return max(0.0, retry_after)

                headers = None
                response = getattr(exc, "response", None)
                if response is not None:
                    headers = getattr(response, "headers", None)
                if headers is None:
                    headers = getattr(exc, "headers", None)
                if headers:
                    retry_after_raw = None
                    try:
                        retry_after_raw = headers.get("retry-after") or headers.get("Retry-After")
                    except Exception:
                        retry_after_raw = None
                    retry_after = _coerce_float(retry_after_raw)
                    if retry_after is not None:
                        return max(0.0, retry_after)

                    reset_raw = None
                    try:
                        reset_raw = headers.get("x-ratelimit-reset-requests") or headers.get("X-RateLimit-Reset-Requests")
                    except Exception:
                        reset_raw = None
                    if reset_raw:
                        reset_text = str(reset_raw).strip().lower()
                        match = re.match(r"^(\\d+(?:\\.\\d+)?)(ms|s)$", reset_text)
                        if match:
                            amount = _coerce_float(match.group(1))
                            if amount is not None:
                                unit = match.group(2)
                                return max(0.0, amount / 1000.0) if unit == "ms" else max(0.0, amount)

                match = re.search(r"retry after\\s+(\\d+(?:\\.\\d+)?)\\s*seconds", msg, flags=re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None
                return None

            async def _acall_with_rate_limit_backoff(*, model: str, input_payload: Any) -> Any:
                attempt = 0
                while True:
                    try:
                        return await self.async_client.embeddings.create(model=model, input=input_payload, **embedding_kwargs)
                    except Exception as exc:  # noqa: BLE001
                        retry_after = _retry_after_seconds(exc)
                        if retry_after is None or attempt >= rate_limit_max_retries:
                            raise
                        sleep_for = max(retry_after, rate_limit_default_sleep) if retry_after else rate_limit_default_sleep
                        sleep_for = min(rate_limit_max_sleep, max(0.0, sleep_for) + random.uniform(0.0, 1.0))
                        logger.warning(
                            "Async embedding rate-limited (429); sleeping %.2fs before retry %d/%d",
                            sleep_for,
                            attempt + 1,
                            rate_limit_max_retries,
                        )
                        await asyncio.sleep(sleep_for)
                        attempt += 1

            if supports_batch_input:
                for i in range(0, len(text_list), batch_size):
                    batch = text_list[i:i + batch_size]
                    response = await _acall_with_rate_limit_backoff(model=self.model_name, input_payload=batch)
                    embeddings.extend(data.embedding for data in response.data)
            else:
                for text in text_list:
                    response = await _acall_with_rate_limit_backoff(model=self.model_name, input_payload=text)
                    embeddings.extend(data.embedding for data in response.data)

            return embeddings[0] if is_single else embeddings

        except Exception as e:
            logger.error(f"Async embedding failed: {str(e)}")
            raise

    # ==================== CONVENIENCE METHODS ====================

    def embed_chunks(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple chunks - always returns list of embeddings"""
        return self.embed(texts) if isinstance(texts, list) else [self.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        """Embed single query - always returns single embedding"""
        result = self.embed(text)
        return result if isinstance(result, list) and isinstance(result[0], (int, float)) else result[0]

    # ==================== UTILITY METHODS ====================

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        """
        # Safely get client info without forcing initialization
        client = getattr(self, 'client', None)

        return {
            "model": self.model_name,
            "api_base": getattr(client, 'base_url', None) if client else None,
            "organization": getattr(client, 'organization', None) if client else None,
            "max_retries": getattr(client, 'max_retries', None) if client else None,
            "timeout": getattr(client, 'timeout', None) if client else None,
            "embedding_dimensions": self.embedding_dimensions,
            "provider": "openai",
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, 'type', 'unknown')
        }

    def _validate_input(self, input_text: str, max_length: Optional[int] = None) -> bool:
        """Validate input text"""
        if not isinstance(input_text, str):
            logger.error("Input must be string type")
            return False

        if not input_text.strip():
            logger.error("Input text cannot be empty")
            return False

        if max_length and len(input_text) > max_length:
            logger.error(f"Input text length exceeds limit: {len(input_text)} > {max_length}")
            return False

        return True
