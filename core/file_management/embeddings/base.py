from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List

class Embeddings(ABC):
    """嵌入接口"""

    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        pass
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.embed_documents, texts
        )
    
    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.embed_query, text
        )

