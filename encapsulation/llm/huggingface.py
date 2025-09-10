from encapsulation.llm.base import LLMBase, LLMBaseConfig
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, TYPE_CHECKING
from pydantic import Field
from dataclasses import dataclass
from framework.shared_module_decorator import shared_module

if TYPE_CHECKING:
    from core.utils.data_model import Document

@shared_module
class HuggingFaceEmbedConfig(LLMBaseConfig):
    """
    HuggingFace embedding model configuration
    """
    type: Literal["huggingface_embedding"] = "huggingface_embedding"
    task_types: Literal["embedding"] = Field(default="embedding", description="Supported task types")


    device: str = Field(default="cpu", description="Device to use for embedding")
    cache_folder: Optional[str] = Field(default=None, description="Cache folder for embedding model")
    model_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Model kwargs for embedding model")
    encode_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Encode kwargs for embedding model")
    
    def build(self, **kwargs) -> "HuggingFaceEmbed":
        """Build the HuggingFace embedding model"""
        return HuggingFaceEmbed(config=self)




@dataclass
class HuggingFaceEmbed(LLMBase[HuggingFaceEmbedConfig]):
    """
    HuggingFace embedding model implementation
    Pure embedding operations - no business logic
    """
    
    config: HuggingFaceEmbedConfig
    
    def __post_init__(self):
        """Initialize after config is set"""
        self._setup_logging()
        self._client = None
        # Initialize embedding model
        self._init_model()
    
    def _init_model(self):
        """Initialize sentence transformer for embedding"""
        try:
            import sentence_transformers
            model_kwargs = self.config.model_kwargs or {}
            self._client = sentence_transformers.SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_folder,
                device=self.config.device,
                **model_kwargs
            )
            
            self.logger.info(f"HuggingFace LLM initialized: {self.config.model_name}")
            
        except ImportError:
            raise ImportError("sentence-transformers required for embedding task")
    
    def _embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        文本嵌入生成
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            嵌入向量或嵌入向量列表
        """
        # Handle single text vs list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        try:
            embeddings = self.embed_documents(text_list)
            return embeddings[0] if is_single else embeddings
        except Exception as e:
            self.logger.error(f"Embedding failed: {str(e)}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            # Clean texts
            texts = [text.replace("\n", " ") for text in texts]
            
            encode_kwargs = self.config.encode_kwargs or {}
            embeddings = self._client.encode(
                texts,
                convert_to_tensor=False,
                **encode_kwargs
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            self.logger.error(f"Document embedding failed: {str(e)}")
            raise RuntimeError(f"Document embedding failed: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.embed_documents([text])[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        info.update({
            "device": self.config.device,
            "cache_folder": self.config.cache_folder,
            "provider": "huggingface",
            "model_type": "sentence_transformer"
        })
        return info
    
    # ==================== NOT SUPPORTED METHODS ====================
    
    def _chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """HuggingFace embedding models don't support chat"""
        raise NotImplementedError("HuggingFace embedding models do not support chat")
    
    def _stream_chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """HuggingFace embedding models don't support streaming chat"""
        raise NotImplementedError("HuggingFace embedding models do not support streaming chat")
    
    def _rerank(self, query: str, documents: List['Document'], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """HuggingFace embedding models don't support reranking"""
        raise NotImplementedError("HuggingFace embedding models do not support reranking")