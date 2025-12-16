from .base import ChatLLMBase
from typing import Any, AsyncGenerator, Dict, List, Optional, TYPE_CHECKING

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
            temperature = float(kwargs.pop("temperature", self.temperature))
            max_tokens = int(kwargs.pop("max_tokens", self.max_tokens))
            return self._hf_generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **kwargs,
            )

        try:
            temperature = kwargs.pop("temperature", self.temperature)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
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
            temperature = kwargs.pop("temperature", self.temperature)
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                # Check for content in choices
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        content = delta.content
                        yield content

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
            temperature = kwargs.pop("temperature", self.temperature)
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
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
            temperature = kwargs.pop("temperature", self.temperature)
            stream = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
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
