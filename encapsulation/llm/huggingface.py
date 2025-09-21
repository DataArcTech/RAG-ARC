from encapsulation.llm.base import LLMBase, LLMBaseConfig
from typing import Union, List, Dict, Any, Optional, Tuple, Literal, TYPE_CHECKING
from pydantic import Field
from dataclasses import dataclass
from framework.shared_module_decorator import shared_module
from framework.module import AbstractModule
import sentence_transformers
import os

if TYPE_CHECKING:
    from core.utils.data_model import Document


class HuggingFaceEmbedConfig(LLMBaseConfig):
    """
    HuggingFace embedding model configuration
    """
    # 修复类型错误：使用正确的类型定义
    type: Literal["huggingface_embedding"] = "huggingface_embedding"
    task_types: Literal["chat", "embedding", "rerank"] = "embedding"


    device: str = Field(default="cpu", description="Device to use for embedding")
    cache_folder: Optional[str] = Field(default=None, description="Cache folder for embedding model")
    model_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Model kwargs for embedding model")
    encode_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Encode kwargs for embedding model")
    # 新增参数：是否使用国内镜像源
    use_china_mirror: bool = Field(default=False, description="Whether to use China mirror for model download")
    # 新增参数：是否只使用本地文件（离线模式）
    local_files_only: bool = Field(default=False, description="Whether to only look at local files (no internet)")
    
    def build(self) -> "HuggingFaceEmbed":
        """Build the HuggingFace embedding model"""
        return HuggingFaceEmbed(self)


@shared_module
class HuggingFaceEmbed(LLMBase[HuggingFaceEmbedConfig]):
    """
    HuggingFace embedding model implementation
    Pure embedding operations - no business logic
    """
    
    config: HuggingFaceEmbedConfig
    
    def __init__(self, config: HuggingFaceEmbedConfig):
        """Initialize after config is set"""
        super().__init__(config=config)
        self.config = config
        self._setup_logging()
        self._client = None
        # Initialize embedding model
        self._init_model()
    
    def _init_model(self):
        """Initialize sentence transformer for embedding"""
        try:
            model_kwargs = self.config.model_kwargs or {}
            
            # 如果使用国内镜像源，设置环境变量
            if self.config.use_china_mirror:
                # 设置HF镜像源
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.logger.info("Set HF_ENDPOINT to https://hf-mirror.com")
            
            # 如果只使用本地文件（离线模式），添加相应参数
            if self.config.local_files_only:
                model_kwargs['local_files_only'] = True
            
            # 注意：SentenceTransformer.__init__ 不接受 mirror 参数
            # 我们通过环境变量 HF_ENDPOINT 来设置镜像源
            self._client = sentence_transformers.SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_folder,
                device=self.config.device,
                **model_kwargs
            )
            
            self.logger.info(f"HuggingFace LLM initialized: {self.config.model_name}")
            if self.config.use_china_mirror:
                self.logger.info("Using China mirror for model download")
            if self.config.local_files_only:
                self.logger.info("Running in offline mode (local files only)")
            
        except ImportError as e:
            self.logger.error(f"sentence-transformers required for embedding task: {str(e)}")
            raise ImportError("sentence-transformers required for embedding task. Please install it with: pip install sentence-transformers") from e
        except Exception as e:
            self.logger.error(f"Failed to initialize HuggingFace model: {str(e)}")
            raise RuntimeError(f"Failed to initialize HuggingFace model: {str(e)}") from e
    
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
            if self._client is not None:
                embeddings = self._client.encode(
                    texts,
                    convert_to_tensor=False,
                    **encode_kwargs
                )
                
                return embeddings.tolist()
            else:
                raise RuntimeError("Model client not initialized")
            
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
            "model_type": "sentence_transformer",
            "use_china_mirror": self.config.use_china_mirror,
            "local_files_only": self.config.local_files_only
        })
        return info
    
    # ==================== NOT SUPPORTED METHODS ====================
    
    def _chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """HuggingFace embedding models don't support chat"""
        raise NotImplementedError("HuggingFace embedding models do not support chat")

    def _achat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs) -> str:
        raise NotImplementedError("HuggingFace embedding models do not support async chat")
    
    def astream_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: Optional[float] = None, **kwargs):
        """HuggingFace embedding models don't support streaming chat"""
        raise NotImplementedError("HuggingFace embedding models do not support streaming chat")
    
    def _stream_chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """HuggingFace embedding models don't support streaming chat"""
        raise NotImplementedError("HuggingFace embedding models do not support streaming chat")
    
    def _rerank(self, query: str, documents: List['Document'], top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """HuggingFace embedding models don't support reranking"""
        raise NotImplementedError("HuggingFace embedding models do not support reranking")