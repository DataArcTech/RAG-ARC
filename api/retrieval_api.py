#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一检索API接口
为上层服务提供简洁、统一的检索接口，支持多种检索器类型
"""

import json
import logging
from typing import Dict, List, Any, Optional, Literal
from pathlib import Path

from core.retrieval.base import BaseRetriever
from config.core.retrieval.dense_config import DenseRetrieverConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from config.core.retrieval.multipath_config import MultiPathRetrieverConfig
from encapsulation.data_model.data_model import Document

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

            # 对于BM25检索器，确保索引已初始化
            if retriever_type == "tantivy_bm25":
                try:
                    # 尝试初始化索引（如果索引目录不存在或为空）
                    index_instance = retriever._index
                    if hasattr(index_instance, '_index') and index_instance._index is None:
                        # 索引未初始化，需要先构建索引才能使用
                        logger.info(f"BM25 index for {name} needs to be built with documents before use")
                except Exception as e:
                    logger.debug(f"BM25 index initialization check failed: {e}")

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
        **kwargs
    ) -> List[Document]:
        """
        执行搜索
        
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
            results = retriever.invoke(query, k=k, **kwargs)
            logger.debug(f"Search completed: {len(results)} results for query '{query}'")
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
    ) -> List[Document]:
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
    
    def add_documents(self, retriever_name: str, documents: List[Document]) -> List[str]:
        """
        添加文档到检索器
        
        Args:
            retriever_name: 检索器名称
            documents: 文档列表
            
        Returns:
            添加的文档ID列表
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        
        try:
            retriever = self.retrievers[retriever_name]
            doc_ids = retriever.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to retriever {retriever_name}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to retriever {retriever_name}: {e}")
            raise
    
    def delete_documents(self, retriever_name: str, ids: List[str]) -> bool:
        """
        删除文档
        
        Args:
            retriever_name: 检索器名称
            ids: 要删除的文档ID列表
            
        Returns:
            删除是否成功
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        
        try:
            retriever = self.retrievers[retriever_name]
            result = retriever.delete_documents(ids)
            logger.info(f"Deleted {len(ids)} documents from retriever {retriever_name}")
            return result is not False
            
        except Exception as e:
            logger.error(f"Failed to delete documents from retriever {retriever_name}: {e}")
            raise
    
    def build_index(self, retriever_name: str, documents: List[Document]) -> None:
        """
        构建索引（仅在索引不存在时使用，如果索引已存在会抛出异常）

        Args:
            retriever_name: 检索器名称
            documents: 文档列表

        Raises:
            RuntimeError: 如果索引已存在
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")

        try:
            retriever = self.retrievers[retriever_name]
            logger.info(f"Building index for {retriever_name}")

            # 直接调用检索器的build_index方法，让检索器自己处理索引存在检查
            retriever.build_index(documents)

            logger.info(f"Successfully built index for retriever {retriever_name} with {len(documents)} documents")

        except RuntimeError as e:
            # 重新抛出RuntimeError（索引已存在的错误）
            logger.warning(f"Cannot build index for retriever {retriever_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to build index for retriever {retriever_name}: {e}")
            raise



    def initialize_index(self, retriever_name: str, documents: List[Document]) -> None:
        """
        初始化索引（强制重新构建，即使索引已存在）

        Args:
            retriever_name: 检索器名称
            documents: 文档列表
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")

        try:
            retriever = self.retrievers[retriever_name]

            logger.info(f"Force initializing index for {retriever_name}")

            # 清空现有索引
            if hasattr(retriever, '_index'):
                index = retriever._index

                # 对于BM25索引器
                if hasattr(index, '_index'):
                    logger.debug(f"Resetting BM25 index for {retriever_name}")
                    index._index = None
                    # 重新初始化BM25索引器
                    if hasattr(index, '_initialize_index'):
                        index._initialize_index()
                    # 确保索引路径存在
                    if hasattr(index, 'config') and hasattr(index.config, 'index_path'):
                        import os
                        os.makedirs(index.config.index_path, exist_ok=True)

                # 对于FAISS索引器
                if hasattr(index, 'index') and hasattr(index, 'docstore'):
                    logger.debug(f"Resetting FAISS index for {retriever_name}")
                    index.index = None
                    index.docstore.clear()
                    index.index_to_docstore_id.clear()

            # 构建新索引
            if hasattr(retriever, '_index') and hasattr(retriever._index, 'from_documents'):
                # 对于BM25索引器，需要特殊处理
                logger.debug(f"Force rebuilding BM25 index for {retriever_name}")

                # 先关闭现有索引
                if hasattr(retriever._index, 'close'):
                    retriever._index.close()

                # 删除现有索引文件
                if hasattr(retriever._index, 'config') and hasattr(retriever._index.config, 'index_path'):
                    import shutil
                    import time
                    index_path = retriever._index.config.index_path
                    if os.path.exists(index_path):
                        try:
                            shutil.rmtree(index_path)
                            logger.debug(f"Removed existing index directory: {index_path}")
                        except OSError as e:
                            logger.warning(f"Failed to remove index directory: {e}, trying again...")
                            time.sleep(0.1)  # 短暂等待
                            try:
                                shutil.rmtree(index_path)
                                logger.debug(f"Successfully removed index directory on retry: {index_path}")
                            except OSError:
                                logger.warning(f"Still failed to remove directory, continuing anyway...")
                    os.makedirs(index_path, exist_ok=True)

                # 重新创建索引器实例（确保没有自动加载）
                retriever._index = retriever.config.index_config.build()

                # 确保索引器状态正确
                retriever._index._index = None
                retriever._index._schema = None
                retriever._index._tokenizers_registered = False

                # 使用from_documents创建索引
                retriever._index.from_documents(documents)

                # 重新加载搜索器
                if hasattr(retriever, 'reload_searcher'):
                    retriever.reload_searcher()
                elif hasattr(retriever, 'searcher'):
                    retriever.searcher = None
            else:
                # 对于其他类型的检索器，使用build_index方法（因为索引已被清空）
                retriever._index.build_index(documents)

            logger.info(f"Successfully initialized index for {retriever_name} with {len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to initialize index for retriever {retriever_name}: {e}")
            raise
    
    def save_index(self, retriever_name: str, index_path: str, index_name: str = "index") -> None:
        """
        保存索引
        
        Args:
            retriever_name: 检索器名称
            index_path: 索引保存路径
            index_name: 索引名称
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        
        try:
            retriever = self.retrievers[retriever_name]
            retriever.save_index(index_path, index_name)
            logger.info(f"Saved index for retriever {retriever_name} to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index for retriever {retriever_name}: {e}")
            raise
    
    def load_index(self, retriever_name: str, index_path: Optional[str] = None) -> None:
        """
        加载索引
        
        Args:
            retriever_name: 检索器名称
            index_path: 索引路径（可选）
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        
        try:
            retriever = self.retrievers[retriever_name]
            retriever.load_index(index_path)
            logger.info(f"Loaded index for retriever {retriever_name}")
            
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
