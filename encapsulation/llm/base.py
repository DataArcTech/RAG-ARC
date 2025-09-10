from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple, Literal, Generic, TypeVar, TYPE_CHECKING
from pydantic import Field
import logging
from dataclasses import dataclass


if TYPE_CHECKING:
    from core.utils.data_model import Document
from framework.config import AbstractConfig
from framework.module import AbstractModule

logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="LLMBaseConfig")

class LLMBaseConfig(AbstractConfig):
    """
    Abstract base class for all LLM configurations.
    - Subclasses must define `type: Literal["xxx"]`
    - Subclasses must implement build() to return the corresponding LLM
    """
    type: Literal["base_llm"] = "base_llm"

    model_name: str = Field(description="Model name")
    task_types: Literal["chat", "embedding", "rerank"] = Field(description="Supported task types")
    kwargs: dict = Field(default_factory=dict, description="Additional configuration parameters")

    @abstractmethod
    def build(self) -> "LLMBase":
        """Build the LLM"""
        raise NotImplementedError("Subclasses must implement build() method")



class LLMBase(AbstractModule, Generic[ConfigType], ABC):
    """
    Unified model base class supporting multiple task types
    Supports: chat, embedding, reranking
    """
    
    config: ConfigType
  
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def supports_task(self, task_type: str) -> bool:
        """Check if specified task type is supported"""
        return task_type in self.config.task_types
    
    def validate_task_support(self, task_type: str):
        """Validate task support, raise exception if not supported"""
        if not self.supports_task(task_type):
            raise ValueError(f"Model {self.config.model_name} does not support task: {task_type}. Supported: {self.config.task_types}")
    
    # ==================== CHAT METHODS ====================
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Chat completion
        """
        self.validate_task_support('chat')
        return self._chat(messages, max_tokens, temperature, **kwargs)
    
    @abstractmethod
    def _chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Internal chat implementation"""
        pass

    def stream_chat(
        self,
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        Streaming chat completion
        """
        self.validate_task_support('chat')
        return self._stream_chat(messages, max_tokens, temperature, **kwargs)
    
    @abstractmethod
    def _stream_chat(
        self,
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """Internal streaming chat implementation"""
        pass
    
    # ==================== EMBEDDING METHODS ====================
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate text embeddings
        """
        self.validate_task_support('embedding')
        return self._embed(texts)
    
    @abstractmethod
    def _embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Internal embedding implementation"""
        pass
    
    # ==================== RERANKING METHODS ====================
    def rerank(
        self, 
        query: str, 
        documents: List['Document'], 
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Document reranking
        
        Args:
            query: Query text
            documents: List of Document objects
            top_k: Return top k results
            
        Returns:
            List of (document_index, score) tuples sorted by score
        """
        self.validate_task_support('rerank')
        return self._rerank(query, documents, top_k)
    
    @abstractmethod 
    def _rerank(
        self, 
        query: str, 
        documents: List['Document'], 
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """Internal reranking implementation"""
        pass
    
    # ==================== UTILITY METHODS ====================
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        """
        return {
            "model_name": self.config.model_name,
            "task_types": self.config.task_types,
            "config": self.config.kwargs,
            "class_name": self.__class__.__name__
        }
    
    def validate_input(self, input_text: str, max_length: Optional[int] = None) -> bool:
        """
        Validate input text
        """
        if not isinstance(input_text, str):
            self.logger.error("Input must be string type")
            return False
        
        if not input_text.strip():
            self.logger.error("Input text cannot be empty")
            return False
        
        if max_length and len(input_text) > max_length:
            self.logger.error(f"Input text length exceeds limit: {len(input_text)} > {max_length}")
            return False
        
        return True
    
    def format_messages(self, user_message: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Format chat messages
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(model_name='{self.config.model_name}', tasks={self.config.task_types})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return f"{self.__class__.__name__}(model_name='{self.config.model_name}', tasks={self.config.task_types}, config={self.config.kwargs})"