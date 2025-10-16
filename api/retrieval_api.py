import json
import logging
from typing import Dict, List, Any, Optional, Literal
from pathlib import Path

from core.retrieval.base import BaseRetriever
from config.core.retrieval.dense_config import DenseRetrieverConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from config.core.retrieval.multipath_config import MultiPathRetrieverConfig
from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)

class RetrievalAPI:
    """
    统一检索API，提供简洁的接口为上层服务使用
    
    支持的检索器类型：
    - dense: 基于向量数据库的密集检索
    - tantivy_bm25: 基于BM25的检索
    - multipath: 多路径融合检索
    """
    
    def __init__(self):
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.configs: Dict[str, Any] = {}
    
    def create_retriever(
        self,
        name: str,
        retriever_type: Literal["dense", "tantivy_bm25", "multipath"],
        config: Dict[str, Any]
    ) -> BaseRetriever:
        """
        创建检索器

        Args:
            name: 检索器名称
            retriever_type: 检索器类型
            config: 配置字典

        Returns:
            创建的检索器实例
        """
        try:
            if retriever_type == "dense":
                retriever_config = DenseRetrieverConfig(**config)
            elif retriever_type == "tantivy_bm25":
                retriever_config = TantivyBM25RetrieverConfig(**config)
            elif retriever_type == "multipath":
                retriever_config = MultiPathRetrieverConfig(**config)
            else:
                raise ValueError(f"Unsupported retriever type: {retriever_type}")

            retriever = retriever_config.build()
            self.retrievers[name] = retriever
            self.configs[name] = config

            logger.info(f"Created {retriever_type} retriever: {name}")
            return retriever

        except Exception as e:
            logger.error(f"Failed to create retriever {name}: {e}")
            raise
    
    def create_from_config_file(self, name: str, config_path: str) -> BaseRetriever:
        """
        从配置文件创建检索器
        
        Args:
            name: 检索器名称
            config_path: 配置文件路径
            
        Returns:
            创建的检索器实例
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            retriever_type = config.get("type")
            if not retriever_type:
                raise ValueError("Config file must specify 'type' field")
            
            return self.create_retriever(name, retriever_type, config)
            
        except Exception as e:
            logger.error(f"Failed to create retriever from config file {config_path}: {e}")
            raise
    
    def search(
        self,
        retriever_name: str,
        query: str,
        k: int = 5,
        owner_id: str = None,
        **kwargs
    ) -> List[Chunk]:
        """
        执行搜索

        Args:
            retriever_name: 检索器名称
            query: 查询文本
            k: 返回结果数量
            owner_id: 用户ID,用于用户隔离检索
            **kwargs: 其他搜索参数

        Returns:
            搜索结果文档列表
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")

        try:
            retriever = self.retrievers[retriever_name]
            # Pass owner_id to retriever for user isolation
            results = retriever.invoke(query, k=k, owner_id=owner_id, **kwargs)
            logger.debug(f"Search completed: {len(results)} results for query '{query}' (owner_id={owner_id})")
            return results

        except Exception as e:
            logger.error(f"Search failed for retriever {retriever_name}: {e}")
            raise
    
    async def asearch(
        self, 
        retriever_name: str, 
        query: str, 
        k: int = 5,
        **kwargs
    ) -> List[Chunk]:
        """
        异步搜索
        
        Args:
            retriever_name: 检索器名称
            query: 查询文本
            k: 返回结果数量
            **kwargs: 其他搜索参数
            
        Returns:
            搜索结果文档列表
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        
        try:
            retriever = self.retrievers[retriever_name]
            results = await retriever.ainvoke(query, k=k, **kwargs)
            logger.debug(f"Async search completed: {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Async search failed for retriever {retriever_name}: {e}")
            raise
    


    def initialize_index(self, retriever_name: str) -> None:
        """
        初始化索引（假设索引已存在，只进行加载操作）

        Args:
            retriever_name: 检索器名称
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")

        try:
            retriever = self.retrievers[retriever_name]
            logger.info(f"Initializing index for {retriever_name}")

            # 获取索引实例
            if hasattr(retriever, '_index'):
                index_instance = retriever._index

                # 如果有load_index方法，尝试加载索引
                if hasattr(index_instance, 'load_index'):
                    # 检查是否需要加载索引
                    needs_loading = False

                    # 对于BM25索引，检查_index是否为None
                    if hasattr(index_instance, '_index') and index_instance._index is None:
                        needs_loading = True
                    # 对于FAISS索引，检查index是否为None
                    elif hasattr(index_instance, 'index') and index_instance.index is None:
                        needs_loading = True
                    # 如果无法判断，默认尝试加载
                    else:
                        needs_loading = True

                    if needs_loading:
                        # 使用配置中的索引路径加载
                        if hasattr(index_instance.config, 'index_path') and index_instance.config.index_path:
                            index_instance.load_index(index_instance.config.index_path)
                        else:
                            index_instance.load_index()
                        logger.info(f"Loaded index for {retriever_name}")

                    # 重新初始化搜索器
                    if hasattr(retriever, 'reload_searcher'):
                        retriever.reload_searcher()
                    elif hasattr(retriever, '_ensure_searcher'):
                        retriever._ensure_searcher()
                else:
                    logger.warning(f"Index for {retriever_name} does not support load_index")
            else:
                logger.warning(f"Retriever {retriever_name} does not have an index instance")

            logger.info(f"Successfully initialized index for {retriever_name}")

        except Exception as e:
            logger.error(f"Failed to initialize index for retriever {retriever_name}: {e}")
            raise
    
    
    def load_index(self, retriever_name: str, index_path: Optional[str] = None) -> None:
        """
        加载索引

        Args:
            retriever_name: 检索器名称
            index_path: 索引路径（可选，如果不提供则使用配置中的路径）
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")

        try:
            retriever = self.retrievers[retriever_name]

            # 获取索引实例
            if hasattr(retriever, '_index'):
                index_instance = retriever._index

                # 调用索引的load_index方法
                if hasattr(index_instance, 'load_index'):
                    # 确定使用的索引路径
                    path_to_use = index_path
                    if not path_to_use and hasattr(index_instance.config, 'index_path'):
                        path_to_use = index_instance.config.index_path

                    # 加载索引
                    if path_to_use:
                        index_instance.load_index(path_to_use)
                        logger.info(f"Loaded index for {retriever_name} from {path_to_use}")
                    else:
                        index_instance.load_index()
                        logger.info(f"Loaded index for {retriever_name}")

                    # 重新初始化搜索器
                    if hasattr(retriever, 'reload_searcher'):
                        retriever.reload_searcher()
                    elif hasattr(retriever, '_ensure_searcher'):
                        retriever._ensure_searcher()
                else:
                    logger.warning(f"Index for {retriever_name} does not support load_index operation")
            else:
                logger.warning(f"Retriever {retriever_name} does not have an index instance")

        except Exception as e:
            logger.error(f"Failed to load index for retriever {retriever_name}: {e}")
            raise
    
    def list_retrievers(self) -> List[str]:
        """获取所有检索器名称列表"""
        return list(self.retrievers.keys())
    
    def get_retriever_info(self, retriever_name: str) -> Dict[str, Any]:
        """
        获取检索器信息
        
        Args:
            retriever_name: 检索器名称
            
        Returns:
            检索器信息字典
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        
        retriever = self.retrievers[retriever_name]
        info = {
            "name": retriever_name,
            "type": retriever.get_name(),
            "class": retriever.__class__.__name__,
            "config": self.configs.get(retriever_name, {})
        }
        
        # 添加特定类型的信息
        if hasattr(retriever, 'get_vectorstore_info'):
            info.update(retriever.get_vectorstore_info())
        elif hasattr(retriever, 'get_multipath_info'):
            info.update(retriever.get_multipath_info())
        
        return info
    
    def remove_retriever(self, retriever_name: str) -> bool:
        """
        移除检索器
        
        Args:
            retriever_name: 检索器名称
            
        Returns:
            移除是否成功
        """
        if retriever_name in self.retrievers:
            del self.retrievers[retriever_name]
            if retriever_name in self.configs:
                del self.configs[retriever_name]
            logger.info(f"Removed retriever: {retriever_name}")
            return True
        return False


# 创建全局API实例
api = RetrievalAPI()
