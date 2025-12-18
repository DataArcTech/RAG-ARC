import asyncio
import time
from typing import Any, AsyncGenerator, Dict, List

from encapsulation.llm.chat.base import ChatLLMBase


class EchoChat(ChatLLMBase):
    def _build_response(self, messages: List[Dict[str, str]]) -> str:
        last = ""
        for msg in reversed(messages or []):
            if msg.get("role") == "user" and (msg.get("content") or "").strip():
                last = msg["content"].strip()
                break
        prefix = getattr(self.config, "prefix", "ECHO:")
        return f"{prefix} {last}".strip()

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        delay_s = float(getattr(self.config, "delay_s", 0.0) or 0.0)
        if delay_s > 0:
            time.sleep(delay_s)
        return self._build_response(messages)

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs: Any):
        full = self._build_response(messages)
        chunk_chars = int(getattr(self.config, "chunk_chars", 16) or 16)
        chunk_chars = max(1, chunk_chars)
        delay_s = float(getattr(self.config, "delay_s", 0.0) or 0.0)
        chunks = [full[i : i + chunk_chars] for i in range(0, len(full), chunk_chars)]
        per_chunk_delay = delay_s / max(len(chunks), 1) if delay_s > 0 else 0.0
        for chunk in chunks:
            if per_chunk_delay > 0:
                time.sleep(per_chunk_delay)
            yield chunk

    async def achat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        delay_s = float(getattr(self.config, "delay_s", 0.0) or 0.0)
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    async def astream_chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]:
        full = await self.achat(messages, **kwargs)
        chunk_chars = int(getattr(self.config, "chunk_chars", 16) or 16)
        chunk_chars = max(1, chunk_chars)
        delay_s = float(getattr(self.config, "delay_s", 0.0) or 0.0)
        chunks = [full[i : i + chunk_chars] for i in range(0, len(full), chunk_chars)]
        per_chunk_delay = delay_s / max(len(chunks), 1) if delay_s > 0 else 0.0
        for chunk in chunks:
            if per_chunk_delay > 0:
                await asyncio.sleep(per_chunk_delay)
            yield chunk

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "echo",
            "prefix": getattr(self.config, "prefix", "ECHO:"),
            "chunk_chars": int(getattr(self.config, "chunk_chars", 16) or 16),
            "delay_s": float(getattr(self.config, "delay_s", 0.0) or 0.0),
        }
