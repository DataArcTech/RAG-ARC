from .base import ChatLLMBase
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from encapsulation.llm.utils.openai_client import create_openai_sync_client, create_openai_async_client
from encapsulation.llm.utils.huggingface_client import create_transformers_client
import logging

logger = logging.getLogger(__name__)


class OpenAIChatLLM(ChatLLMBase):
    """
    OpenAI Chat LLM implementation supporting conversational AI capabilities.

    This class provides a complete chat solution using OpenAI's API,
    supporting multiple model types with streaming and async capabilities.
    Optimized for conversational AI with configurable parameters and retry logic.

    Key features:
    - Multiple model support: GPT-4, GPT-3.5, GPT-4o variants
    - Streaming chat with token usage tracking
    - Async support for high-throughput applications
    - Configurable parameters: temperature, max_tokens
    - Automatic retry logic and timeout handling
    - Flexible API endpoint configuration (OpenAI or compatible)

    Configuration options:
        - api_key: OpenAI API key or compatible service key
        - base_url: API endpoint (defaults to OpenAI, supports custom servers)
        - model_name: Target model identifier
        - max_tokens: Maximum response length
        - temperature: Response creativity (0.0-2.0)
        - organization: OpenAI organization ID
        - timeout: Request timeout in seconds
        - max_retries: Automatic retry attempts
    """

    def __init__(self, config: "OpenAIChatConfig"):
        """Initialize OpenAI Chat with loading method support"""
        super().__init__(config)
        # Cache config values to avoid repeated getattr calls
        self.model_name = getattr(self.config, 'model_name', 'gpt-4o-mini')
        self.max_tokens = getattr(self.config, 'max_tokens', 2000)
        self.temperature = getattr(self.config, 'temperature', 0.1)
        self.loading_method = getattr(self.config, 'loading_method', 'openai')

        # Initialize client based on loading method
        if self.loading_method == 'openai':
            self.client = create_openai_sync_client(self.config)
            self.async_client = create_openai_async_client(self.config)
            self.tokenizer = None
        elif self.loading_method == 'huggingface':
            # For HuggingFace transformers, we get (model, tokenizer) tuple
            self.client, self.tokenizer = create_transformers_client(self.config)
            self.async_client = None  # HuggingFace uses asyncio.to_thread wrapper
        else:
            raise ValueError(f"Unsupported loading method: {self.loading_method}")

    # ==================== CHAT IMPLEMENTATION ====================

    def _build_hf_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style chat messages into a single prompt for HF generation."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001
                pass

        lines: List[str] = []
        for message in messages:
            role = (message.get("role") or "user").strip()
            content = (message.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines).strip()

    def _hf_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> str:
        """Generate a completion using a locally loaded Transformers model."""
        import torch

        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("HuggingFace tokenizer is not initialized")

        model = getattr(self, "client", None)
        if model is None or not hasattr(model, "generate"):
            raise RuntimeError("HuggingFace model is not initialized")

        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = temperature is not None and float(temperature) > 0.0

        supported_kwargs = {
            "top_p",
            "top_k",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "num_beams",
            "eos_token_id",
            "pad_token_id",
        }
        gen_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}

        gen_kwargs.setdefault("pad_token_id", getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None))
        gen_kwargs.setdefault("eos_token_id", getattr(tokenizer, "eos_token_id", None))

        with torch.no_grad():
            generation_params: Dict[str, Any] = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                **gen_kwargs,
            }
            if do_sample:
                generation_params["temperature"] = float(temperature)
            output_ids = model.generate(**generation_params)

        input_len = int(inputs["input_ids"].shape[-1])
        generated = output_ids[0][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """Internal chat implementation"""
        self._validate_messages(messages)

        if self.loading_method == "huggingface":
            prompt = self._build_hf_prompt(messages)
            kwargs.pop("model", None)
            temperature = float(kwargs.pop("temperature", self.temperature))
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            return self._hf_generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        def _extract_provider_error(payload: Any) -> Optional[str]:
            """Detect error-shaped JSON bodies from OpenAI-compatible gateways.

            Some gateways may return HTTP 200 with an error payload (no `choices`),
            which the OpenAI SDK can parse without raising. Surface that as an
            exception so higher-level retry/budget logic can react (e.g., context
            length vs rate limit).
            """

            if not isinstance(payload, dict):
                return None
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("error") or err.get("type") or ""
                code = err.get("code") or err.get("status") or ""
                msg = str(msg).strip()
                code = str(code).strip()
                if code and msg:
                    return f"{code}: {msg}"
                if msg:
                    return msg
                return json.dumps(err, ensure_ascii=False, default=str)[:2000]
            if isinstance(err, str) and err.strip():
                return err.strip()
            if "choices" not in payload:
                msg = payload.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            return None

        try:
            model = kwargs.pop("model", self.model_name)
            temperature = float(kwargs.pop("temperature", self.temperature))
            temperature = self._adjust_temperature_for_model(model, temperature)
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            raw = self.client.with_raw_response.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            try:
                body = raw.http_response.json()
            except Exception:  # noqa: BLE001
                body = {"_raw_text": (raw.http_response.text or "")[:4000]}

            provider_err = _extract_provider_error(body)
            if provider_err:
                raise RuntimeError(f"OpenAI-compatible chat error: {provider_err}")
            if isinstance(body, dict) and "choices" not in body:
                raise RuntimeError(f"OpenAI-compatible chat error: missing choices; keys={sorted(body.keys())!r}")

            response = raw.parse()
            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("OpenAI-compatible chat error: response.choices is empty/None")
            content = getattr(choices[0].message, "content", None)
            if content is None:
                return ""
            return str(content).strip()

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ):
        """
        Internal streaming chat implementation
        """
        self._validate_messages(messages)

        if self.loading_method == "huggingface":
            prompt = self._build_hf_prompt(messages)
            kwargs.pop("model", None)
            temperature = float(kwargs.pop("temperature", self.temperature))
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            try:
                from transformers import TextIteratorStreamer
                import threading
                import inspect

                tokenizer = getattr(self, "tokenizer", None)
                model = getattr(self, "client", None)
                if tokenizer is None or model is None:
                    raise RuntimeError("HuggingFace model/tokenizer is not initialized")

                if "streamer" not in inspect.signature(model.generate).parameters:
                    yield self._hf_generate(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs,
                    )
                    return

                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                import torch

                inputs = tokenizer(prompt, return_tensors="pt")
                device = getattr(model, "device", None)
                if device is not None:
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                do_sample = temperature is not None and float(temperature) > 0.0
                supported_kwargs = {
                    "top_p",
                    "top_k",
                    "repetition_penalty",
                    "no_repeat_ngram_size",
                    "num_beams",
                    "eos_token_id",
                    "pad_token_id",
                }
                gen_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}
                gen_kwargs.setdefault("pad_token_id", getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None))
                gen_kwargs.setdefault("eos_token_id", getattr(tokenizer, "eos_token_id", None))

                def _run_generation():
                    with torch.no_grad():
                        generation_params: Dict[str, Any] = {
                            **inputs,
                            "max_new_tokens": max_tokens,
                            "do_sample": do_sample,
                            "streamer": streamer,
                            **gen_kwargs,
                        }
                        if do_sample:
                            generation_params["temperature"] = float(temperature)
                        model.generate(**generation_params)

                thread = threading.Thread(target=_run_generation, daemon=True)
                thread.start()
                for text in streamer:
                    if text:
                        yield text
                thread.join(timeout=0.1)
                return
            except Exception as e:
                logger.warning("HuggingFace streaming unavailable, falling back to non-streaming: %s", e)
                yield self._hf_generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return

        try:
            model = kwargs.pop("model", self.model_name)
            temperature = float(kwargs.pop("temperature", self.temperature))
            temperature = self._adjust_temperature_for_model(model, temperature)
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            
            # Log complete API request details
            try:
                # Log full messages (truncated per message but all messages included)
                messages_log = []
                for idx, msg in enumerate(messages):
                    msg_role = msg.get("role", "unknown")
                    msg_content = msg.get("content", "")
                    msg_len = len(msg_content)
                    # Log first 500 chars of each message, or full if shorter
                    msg_preview = msg_content[:500] + f"...[truncated {msg_len} chars]" if msg_len > 500 else msg_content
                    messages_log.append({
                        "index": idx,
                        "role": msg_role,
                        "content_length": msg_len,
                        "content_preview": msg_preview
                    })
                
                # Log all request parameters
                request_params = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                    "messages_count": len(messages),
                    "additional_kwargs": kwargs,
                }
                
                logger.info(
                    "OpenAIChatLLM.stream_chat REQUEST: params=%s messages_detail=%s",
                    json.dumps(request_params, ensure_ascii=False, default=str),
                    json.dumps(messages_log, ensure_ascii=False, default=str),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAIChatLLM.stream_chat failed to log request details: %s", exc)
            
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                    **kwargs
                )
            except Exception as exc:
                logger.error(
                    "OpenAIChatLLM.stream_chat API call FAILED: model=%s error=%s error_type=%s",
                    model,
                    str(exc),
                    type(exc).__name__,
                    exc_info=True,
                )
                raise

            chunk_count = 0
            total_content_length = 0
            all_chunks_detail = []
            captured_pieces: list[str] = []
            
            try:
                for chunk in stream:
                    chunk_count += 1
                    # Log COMPLETE chunk structure
                    try:
                        chunk_dict = {
                            'chunk_number': chunk_count,
                            'id': getattr(chunk, 'id', None),
                            'object': getattr(chunk, 'object', None),
                            'created': getattr(chunk, 'created', None),
                            'model': getattr(chunk, 'model', None),
                            'system_fingerprint': getattr(chunk, 'system_fingerprint', None),
                        }
                        
                        # Log choices in detail
                        choices_detail = []
                        if chunk.choices and len(chunk.choices) > 0:
                            for choice_idx, choice in enumerate(chunk.choices):
                                choice_dict = {
                                    'index': getattr(choice, 'index', choice_idx),
                                    'finish_reason': getattr(choice, 'finish_reason', None),
                                    'logprobs': getattr(choice, 'logprobs', None),
                                }
                                
                                # Log delta in COMPLETE detail
                                delta = getattr(choice, 'delta', None)
                                if delta:
                                    delta_dict = {
                                        'role': getattr(delta, 'role', None),
                                        'content': getattr(delta, 'content', None),
                                        'content_type': type(getattr(delta, 'content', None)).__name__ if hasattr(delta, 'content') else None,
                                        'content_repr': repr(getattr(delta, 'content', None)) if hasattr(delta, 'content') else None,
                                        'tool_calls': getattr(delta, 'tool_calls', None),
                                        'refusal': getattr(delta, 'refusal', None),
                                    }
                                    # Skip other_attrs to reduce log size (only log essential fields)
                                    choice_dict['delta'] = delta_dict
                                
                                choices_detail.append(choice_dict)
                        
                        chunk_dict['choices'] = choices_detail
                        chunk_dict['choices_count'] = len(chunk.choices) if chunk.choices else 0
                        
                        # Log usage if present
                        if hasattr(chunk, 'usage') and chunk.usage:
                            chunk_dict['usage'] = {
                                'prompt_tokens': getattr(chunk.usage, 'prompt_tokens', None),
                                'completion_tokens': getattr(chunk.usage, 'completion_tokens', None),
                                'total_tokens': getattr(chunk.usage, 'total_tokens', None),
                            }
                        
                        # Store chunk detail
                        all_chunks_detail.append(chunk_dict)
                        
                        # Log first 3 chunks in FULL detail, then every 20th chunk
                        if chunk_count <= 3 or chunk_count % 20 == 0:
                            logger.info(
                                "OpenAIChatLLM.stream_chat CHUNK %d DETAIL: %s",
                                chunk_count,
                                json.dumps(chunk_dict, ensure_ascii=False, default=str),
                            )
                        
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "OpenAIChatLLM.stream_chat failed to log chunk %d: error=%s chunk_repr=%s",
                            chunk_count,
                            str(exc),
                            repr(chunk)[:500],
                            exc_info=True,
                        )
                    
                    # Check for content in choices
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content is not None:
                            content = delta.content
                            content_len = len(content) if isinstance(content, str) else 0
                            total_content_length += content_len
                            if chunk_count <= 5:  # Log first 5 content pieces
                                logger.debug(
                                    "OpenAIChatLLM.stream_chat chunk %d CONTENT: length=%d content=%s",
                                    chunk_count,
                                    content_len,
                                    repr(content)[:200],
                                )
                            if isinstance(content, str) and content:
                                captured_pieces.append(content)
                            yield content
                        elif chunk_count <= 5:  # Log first 5 chunks even if no content
                            logger.debug(
                                "OpenAIChatLLM.stream_chat chunk %d NO CONTENT: delta_has_content=%s delta_content=%s",
                                chunk_count,
                                hasattr(delta, 'content') if delta else False,
                                getattr(delta, 'content', None) if delta and hasattr(delta, 'content') else None,
                            )
                            
            except Exception as stream_exc:
                logger.error(
                    "OpenAIChatLLM.stream_chat STREAM ERROR: chunk_count=%d total_content_length=%d error=%s error_type=%s",
                    chunk_count,
                    total_content_length,
                    str(stream_exc),
                    type(stream_exc).__name__,
                    exc_info=True,
                )
                raise

            if chunk_count > 0 and total_content_length == 0:
                logger.warning(
                    "OpenAIChatLLM.stream_chat produced 0 content tokens; falling back to non-streaming completion (model=%s).",
                    model,
                )
                try:
                    text = self.chat(
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "OpenAIChatLLM.stream_chat fallback chat() failed: model=%s error=%s",
                        model,
                        exc,
                        exc_info=True,
                    )
                    raise
                if isinstance(text, str) and text:
                    yield text

            # Log COMPLETE summary after streaming completes
            try:
                summary = {
                    'model': model,
                    'total_chunks': chunk_count,
                    'total_content_length': total_content_length,
                    'chunks_with_content': sum(1 for c in all_chunks_detail if c.get('choices') and any(
                        choice.get('delta', {}).get('content') not in (None, '') 
                        for choice in c.get('choices', [])
                    )),
                    'chunks_with_finish_reason': sum(1 for c in all_chunks_detail if c.get('choices') and any(
                        choice.get('finish_reason') is not None 
                        for choice in c.get('choices', [])
                    )),
                    'all_finish_reasons': [
                        choice.get('finish_reason')
                        for c in all_chunks_detail
                        for choice in c.get('choices', [])
                        if choice.get('finish_reason') is not None
                    ],
                    'chunks_detail_summary': [
                        {
                            'chunk_number': c.get('chunk_number'),
                            'has_content': any(
                                choice.get('delta', {}).get('content') not in (None, '')
                                for choice in c.get('choices', [])
                            ),
                            'finish_reason': next((
                                choice.get('finish_reason')
                                for choice in c.get('choices', [])
                                if choice.get('finish_reason') is not None
                            ), None),
                        }
                        for c in all_chunks_detail
                        # Only include chunks with issues (no content or non-stop finish_reason) or first/last 5
                        if not any(
                            choice.get('delta', {}).get('content') not in (None, '')
                            for choice in c.get('choices', [])
                        ) or c.get('chunk_number', 0) <= 5 or c.get('chunk_number', 0) > chunk_count - 5
                    ],
                }
                logger.info(
                    "OpenAIChatLLM.stream_chat COMPLETED SUMMARY: %s",
                    json.dumps(summary, ensure_ascii=False, default=str, indent=2),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OpenAIChatLLM.stream_chat failed to log summary: %s",
                    exc,
                    exc_info=True,
                )

        except Exception as e:
            logger.error(f"Streaming chat failed: {str(e)}")
            raise

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        Async function of chat
        """
        self._validate_messages(messages)

        if self.loading_method == "huggingface":
            import asyncio
            return await asyncio.to_thread(self.chat, messages, **kwargs)

        def _extract_provider_error(payload: Any) -> Optional[str]:
            if not isinstance(payload, dict):
                return None
            err = payload.get("error")
            if isinstance(err, dict):
                msg = err.get("message") or err.get("error") or err.get("type") or ""
                code = err.get("code") or err.get("status") or ""
                msg = str(msg).strip()
                code = str(code).strip()
                if code and msg:
                    return f"{code}: {msg}"
                if msg:
                    return msg
                return json.dumps(err, ensure_ascii=False, default=str)[:2000]
            if isinstance(err, str) and err.strip():
                return err.strip()
            if "choices" not in payload:
                msg = payload.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
            return None

        try:
            model = kwargs.pop("model", self.model_name)
            temperature = float(kwargs.pop("temperature", self.temperature))
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            raw = await self.async_client.with_raw_response.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            try:
                body = raw.http_response.json()
            except Exception:  # noqa: BLE001
                body = {"_raw_text": (raw.http_response.text or "")[:4000]}

            provider_err = _extract_provider_error(body)
            if provider_err:
                raise RuntimeError(f"OpenAI-compatible chat error: {provider_err}")
            if isinstance(body, dict) and "choices" not in body:
                raise RuntimeError(f"OpenAI-compatible chat error: missing choices; keys={sorted(body.keys())!r}")

            # NOTE: `with_raw_response` currently returns a LegacyAPIResponse wrapper in
            # openai-python 2.x (Stainless legacy wrapper). For the async client, its
            # `.parse()` is still synchronous. Do NOT await it.
            response = raw.parse()
            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("OpenAI-compatible chat error: response.choices is empty/None")
            content = getattr(choices[0].message, "content", None)
            if content is None:
                return ""
            return str(content)

        except Exception as e:
            logger.error(f"Async chat failed: {str(e)}")
            raise

    async def astream_chat(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Async function of stream chat
        """
        self._validate_messages(messages)

        if self.loading_method == "huggingface":
            import asyncio
            import threading

            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

            def _runner():
                try:
                    for chunk in self.stream_chat(messages, **kwargs):
                        asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
                finally:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop)

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            return

        try:
            model = kwargs.pop("model", self.model_name)
            temperature = float(kwargs.pop("temperature", self.temperature))
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Async streaming chat failed: {str(e)}")
            raise

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
            "default_max_tokens": self.max_tokens,
            "default_temperature": self.temperature,
            "provider": "huggingface" if self.loading_method == "huggingface" else "openai",
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, 'type', 'unknown')
        }

    def _adjust_temperature_for_model(self, model: str, temperature: float) -> float:
        """
        Adjust temperature for models that only support specific values.
        gpt-5 series models only support temperature=1 (default).
        """
        if model.startswith("gpt-5") and temperature != 1.0:
            logger.info("gpt-5 series model detected (%s), adjusting temperature from %s to 1.0", model, temperature)
            return 1.0
        return temperature

    def _validate_messages(self, messages):
        """Validate message format"""
        if not messages or not isinstance(messages, list):
            raise ValueError("Messages must be a non-empty list")
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("Message format error: must contain 'role' and 'content'")
            if self.loading_method == "huggingface" and not isinstance(msg.get("content"), str):
                raise ValueError("HuggingFace chat only supports string content (no multimodal parts).")
            if not self._validate_input(msg.get('content')):
                raise ValueError(f"Message content validation failed: {msg.get('content')}")

    def _validate_input(self, input_text: Any, max_length: Optional[int] = None) -> bool:
        """Validate input text"""
        if isinstance(input_text, str):
            if not input_text.strip():
                logger.error("Input text cannot be empty")
                return False
            if max_length and len(input_text) > max_length:
                logger.error("Input text length exceeds limit: %s > %s", len(input_text), max_length)
                return False
            return True

        # OpenAI-compatible multimodal: content is a list of parts.
        if isinstance(input_text, list):
            if not input_text:
                logger.error("Input parts cannot be empty")
                return False
            for part in input_text:
                if not isinstance(part, dict) or "type" not in part:
                    logger.error("Invalid multimodal part: %r", part)
                    return False
                part_type = part.get("type")
                if part_type == "text":
                    text = part.get("text")
                    if not isinstance(text, str) or not text.strip():
                        logger.error("Invalid text part: %r", part)
                        return False
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    url = None
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                    if not isinstance(url, str) or not url.strip():
                        logger.error("Invalid image_url part: %r", part)
                        return False
                else:
                    logger.error("Unsupported multimodal part type: %s", part_type)
                    return False
            return True

        logger.error("Input must be string or multimodal part list")
        return False
