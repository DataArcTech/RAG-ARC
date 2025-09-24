from abc import abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
import logging

from framework.module import AbstractModule

logger = logging.getLogger(__name__)


class ChatLLMBase(AbstractModule):
    """
    Base class for chat LLM implementations
    Supports conversational AI with sync/async and streaming capabilities
    """

    # ==================== CHAT METHODS ====================
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Chat completion"""
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ):
        """Streaming chat completion"""
        pass

    # ==================== ASYNC CHAT METHODS ====================
    @abstractmethod
    async def achat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Async chat completion"""
        pass

    @abstractmethod
    async def astream_chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat completion"""
        pass

    # ==================== UTILITY METHODS ====================
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass