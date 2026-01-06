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
        self.temperature = getattr(self.config, 'temperature', 0.7)
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
        messages: List[Dict[str, str]],
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

        try:
            model = kwargs.pop("model", self.model_name)
            temperature = float(kwargs.pop("temperature", self.temperature))
            temperature = self._adjust_temperature_for_model(model, temperature)
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            result = response.choices[0].message.content.strip()
            return result

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
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
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Async function of chat
        """
        self._validate_messages(messages)

        if self.loading_method == "huggingface":
            import asyncio
            return await asyncio.to_thread(self.chat, messages, **kwargs)

        try:
            model = kwargs.pop("model", self.model_name)
            temperature = float(kwargs.pop("temperature", self.temperature))
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            result = response.choices[0].message.content
            return result

        except Exception as e:
            logger.error(f"Async chat failed: {str(e)}")
            raise

    async def astream_chat(
        self,
        messages: List[Dict[str, str]],
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
            if not self._validate_input(msg['content']):
                raise ValueError(f"Message content validation failed: {msg['content']}")

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
