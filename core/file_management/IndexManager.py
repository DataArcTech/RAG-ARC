"""
IndexManager - 统一索引管理器

负责从关系数据库中读取未构建索引的chunk数据，并调用相应的索引器进行索引构建。
该设计将索引构建逻辑从Retriever中分离，实现了索引管理的集中化。
"""

import logging
import json
from typing import List, Dict, Any, Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pydantic import Field

from framework.module import AbstractModule
from framework.config import AbstractConfig

from encapsulation.database.relational_db.data_schema import ChunksStatus
from core.utils.data_model import Document

logger = logging.getLogger(__name__)



class IndexManagerConfig(AbstractConfig):
    """IndexManager 配置类
    
    统一索引管理器的配置，负责管理多种索引类型的构建和维护。
    """
    type: Literal["index_manager"] = "index_manager"
    
    # 数据库配置
    relational_db_config: Dict[str, Any] = Field(
        description="关系数据库配置，用于存储chunks元数据"
    )
    file_db_config: Dict[str, Any] = Field(
        description="文件数据库配置，用于存储chunk数据"
    )
    
    # 索引器配置映射
    indexer_configs: Dict[str, Dict[str, Any]] = Field(
        description="索引器配置映射，key为index_type，value为对应的索引器配置"
    )

    # 嵌入模型配置（用于FAISS索引）
    embedding_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="嵌入模型配置，用于FAISS索引的向量计算"
    )
    
    # 批处理配置
    batch_size: int = Field(
        default=100, 
        description="批处理大小，每次处理的chunk数量"
    )
    max_concurrent_builds: int = Field(
        default=3, 
        description="最大并发构建数，同时构建的索引类型数量"
    )
    
    # 重试配置
    max_retries: int = Field(
        default=3, 
        description="最大重试次数"
    )
    retry_delay: float = Field(
        default=1.0, 
        description="重试延迟（秒）"
    )
    
    def build(self) -> "IndexManager":
        return IndexManager(self)



class IndexManager(AbstractModule):
    """
    统一索引管理器

    负责从数据库读取未构建索引的chunk数据，调用相应索引器进行构建，
    并更新元数据状态。支持多种索引类型的统一管理。

    核心功能：
    - 索引构建：从CHUNKED状态的chunks构建索引
    - 索引重建：强制重建指定类型的索引
    - 状态管理：更新chunks的索引状态
    - 统计信息：获取各类型索引的统计信息
    """

    def __init__(self, config: IndexManagerConfig):
        super().__init__(config)
        self._relational_db = None
        self._file_db = None
        self._indexers: Dict[str, Any] = {}
        self._embedding_model = None
        self._initialize_dependencies()

    def _initialize_dependencies(self):
        """初始化依赖组件"""
        # 初始化数据库连接
        self._initialize_databases()
        # 初始化嵌入模型（如果配置了）
        self._initialize_embedding_model()
        # 初始化索引器实例
        self._initialize_indexers()

    def _initialize_databases(self):
        """初始化数据库连接"""
        try:
            from encapsulation.database.relational_db.postgresql import PostgreSQLDB
            from encapsulation.database.file_db.local import LocalDB

            # 创建配置对象
            relational_db_config = self._create_config_object(
                self.config.relational_db_config, "relational_db"
            )
            file_db_config = self._create_config_object(
                self.config.file_db_config, "file_db"
            )

            self._relational_db = PostgreSQLDB(relational_db_config)
            self._file_db = LocalDB(file_db_config)

            logger.info("Database connections initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            raise

    def _create_config_object(self, config_dict: Dict[str, Any], config_type: str):
        """根据配置字典创建配置对象"""
        if isinstance(config_dict, dict):
            config_type_value = config_dict.get("type", "")

            if config_type == "relational_db":
                if config_type_value == "postgresql":
                    from framework.config import AbstractConfig
                    from typing import Literal

                    class PostgreSQLConfig(AbstractConfig):
                        type: Literal["postgresql"] = "postgresql"
                        host: str = "localhost"
                        port: int = 5432
                        database: str
                        user: str
                        password: str

                        def build(self):
                            from encapsulation.database.relational_db.postgresql import PostgreSQLDB
                            return PostgreSQLDB(self)

                    return PostgreSQLConfig(**config_dict)

            elif config_type == "file_db":
                if config_type_value == "local":
                    from framework.config import AbstractConfig
                    from typing import Literal

                    class LocalDBConfig(AbstractConfig):
                        type: Literal["local"] = "local"
                        base_path: str
                        cleanup_empty_dirs: bool = True

                        def build(self):
                            from encapsulation.database.file_db.local import LocalDB
                            return LocalDB(self)

                    return LocalDBConfig(**config_dict)

        # 如果已经是配置对象，直接返回
        return config_dict

    def _initialize_embedding_model(self):
        """初始化嵌入模型（用于FAISS索引）"""
        if self.config.embedding_config is None:
            logger.info("No embedding model configured, FAISS indexing will be skipped")
            return

        try:
            embedding_type = self.config.embedding_config.get("type", "")

            if embedding_type == "huggingface_embedding":
                from encapsulation.llm.huggingface import HuggingFaceEmbedConfig, HuggingFaceEmbed

                # 创建嵌入模型配置
                embed_config = HuggingFaceEmbedConfig(**self.config.embedding_config)
                self._embedding_model = HuggingFaceEmbed(embed_config)
                logger.info(f"Initialized HuggingFace embedding model: {embed_config.model_name}")

            elif embedding_type == "openai":
                from encapsulation.llm.openai import OpenAIConfig, OpenAILLM

                # 创建OpenAI嵌入模型配置
                embed_config = OpenAIConfig(**self.config.embedding_config)
                self._embedding_model = OpenAILLM(embed_config)
                logger.info(f"Initialized OpenAI embedding model: {embed_config.model_name}")

            else:
                logger.warning(f"Unsupported embedding type: {embedding_type}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # 不抛出异常，允许继续运行但跳过FAISS索引

    def _initialize_indexers(self):
        """根据配置初始化各类型索引器"""
        for index_type, indexer_config_dict in self.config.indexer_configs.items():
            try:
                if index_type == "faiss":
                    from encapsulation.database.vector_db.faiss import FaissIndexConfig, FaissIndex
                    config = FaissIndexConfig(**indexer_config_dict)
                    self._indexers[index_type] = FaissIndex(config)
                    logger.info(f"Initialized FAISS indexer: {config.index_path}")
                elif index_type == "bm25_indexer":
                    from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig, BM25IndexBuilder
                    config = BM25IndexBuilderConfig(**indexer_config_dict)
                    self._indexers[index_type] = BM25IndexBuilder(config)
                    logger.info(f"Initialized BM25 indexer: {config.index_path}")
                else:
                    logger.warning(f"Unsupported index type: {index_type}")
            except Exception as e:
                logger.error(f"Failed to initialize indexer {index_type}: {e}")
                # 不抛出异常，允许部分索引器初始化失败

    def build_pending_indexes(self) -> Dict[str, int]:
        """
        构建所有待处理的索引

        Returns:
            Dict[str, int]: 每种索引类型构建的数量统计
        """
        # 获取状态为CHUNKED的chunks元数据
        unindexed_chunks = self._relational_db.list_chunks_metadata(
            status=ChunksStatus.CHUNKED
        )

        if not unindexed_chunks:
            logger.info("No pending chunks to index")
            return {}

        logger.info(f"Found {len(unindexed_chunks)} chunks pending indexing")

        # 按index_type分组
        chunks_by_type = self._group_chunks_by_index_type(unindexed_chunks)
        build_stats = {}

        # 并发构建不同类型的索引
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_builds) as executor:
            futures = {}
            for index_type, chunks in chunks_by_type.items():
                if index_type in self._indexers:
                    future = executor.submit(self._build_index_for_type, index_type, chunks)
                    futures[future] = index_type
                else:
                    logger.warning(f"No indexer available for type: {index_type}")
                    build_stats[index_type] = 0

            # 收集结果
            for future in as_completed(futures):
                index_type = futures[future]
                try:
                    count = future.result()
                    build_stats[index_type] = count
                    logger.info(f"Built {count} indexes for type: {index_type}")
                except Exception as e:
                    logger.error(f"Failed to build indexes for type {index_type}: {e}")
                    build_stats[index_type] = 0

        return build_stats

    def _group_chunks_by_index_type(self, chunks_metadata: List[Any]) -> Dict[str, List[Any]]:
        """按索引类型分组chunks"""
        groups = {}
        for chunk_meta in chunks_metadata:
            # 根据配置或chunk元数据确定index_type
            index_type = self._determine_index_type(chunk_meta)
            if index_type not in groups:
                groups[index_type] = []
            groups[index_type].append(chunk_meta)
        return groups

    def _determine_index_type(self, chunk_meta: Any) -> str:
        """确定chunk应使用的索引类型"""
        # 可以根据chunk的特征、配置等决定使用哪种索引
        # 这里简化为默认策略
        if hasattr(chunk_meta, 'index_type') and chunk_meta.index_type:
            return chunk_meta.index_type

        # 默认策略：使用第一个可用的索引类型
        available_types = list(self._indexers.keys())
        if available_types:
            return available_types[0]

        # 如果没有可用的索引器，返回默认值
        return "faiss"

    def _build_index_for_type(self, index_type: str, chunks_metadata: List[Any]) -> int:
        """为特定类型构建索引"""
        indexer = self._indexers[index_type]
        built_count = 0

        # 批处理构建
        for i in range(0, len(chunks_metadata), self.config.batch_size):
            batch = chunks_metadata[i:i + self.config.batch_size]
            retry_count = 0

            while retry_count <= self.config.max_retries:
                try:
                    # 加载chunk数据
                    documents = self._load_documents_from_chunks(batch)
                    if not documents:
                        logger.warning(f"No documents loaded from batch {i//self.config.batch_size + 1}")
                        break

                    # 构建索引
                    if hasattr(indexer, 'add'):
                        # 对于需要嵌入向量的索引（如FAISS），需要计算嵌入向量
                        if index_type == "faiss":
                            if self._embedding_model is None:
                                logger.warning(f"FAISS indexer requires embedding model, but none configured. Skipping.")
                                break

                            # 计算嵌入向量
                            texts = [doc.content for doc in documents]
                            embeddings = self._embedding_model.embed_documents(texts)
                            indexer.add(documents, embeddings)
                        else:
                            indexer.add(documents)
                    elif hasattr(indexer, 'from_documents'):
                        indexer.from_documents(documents)
                    else:
                        logger.error(f"Indexer {index_type} does not support add or from_documents")
                        break

                    # 更新元数据状态
                    self._update_chunks_status(batch, ChunksStatus.INDEXED, index_type)
                    built_count += len(batch)
                    logger.info(f"Built index for {len(batch)} chunks of type {index_type}")
                    break  # 成功，跳出重试循环

                except Exception as e:
                    retry_count += 1
                    if retry_count <= self.config.max_retries:
                        logger.warning(f"Failed to build index batch for type {index_type} (attempt {retry_count}): {e}")
                        import time
                        time.sleep(self.config.retry_delay)
                    else:
                        logger.error(f"Failed to build index batch for type {index_type} after {self.config.max_retries} retries: {e}")
                        # 标记失败状态
                        self._update_chunks_status(batch, ChunksStatus.FAILED, index_type)

        return built_count

    def _load_documents_from_chunks(self, chunks_metadata: List[Any]) -> List[Document]:
        """从chunks元数据加载Document对象"""
        documents = []
        for chunk_meta in chunks_metadata:
            try:
                # 从文件数据库加载chunk数据
                chunk_data = self._file_db.retrieve(chunk_meta.blob_key)

                # 解析JSON数据为Document对象
                chunks_json = json.loads(chunk_data.decode('utf-8'))
                for chunk in chunks_json.get('chunks', []):
                    doc = Document(
                        id=chunk.get('id'),
                        content=chunk.get('content'),
                        metadata={
                            **chunk.get('metadata', {}),
                            'chunks_id': chunk_meta.chunks_id,
                            'source_parsed_content_id': chunk_meta.source_parsed_content_id
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load documents from chunk {chunk_meta.chunks_id}: {e}")

        return documents

    def _update_chunks_status(self, chunks_metadata: List[Any], status: ChunksStatus, index_type: str):
        """更新chunks状态"""
        for chunk_meta in chunks_metadata:
            try:
                # 更新状态和索引类型
                updates = {
                    "status": status,
                    "index_type": index_type
                }

                # 保存到数据库
                success = self._relational_db.update_chunks_metadata(chunk_meta.chunks_id, updates)
                if not success:
                    logger.warning(f"Failed to update chunk status {chunk_meta.chunks_id}")
            except Exception as e:
                logger.error(f"Failed to update chunk status {chunk_meta.chunks_id}: {e}")

    def rebuild_index(self, index_type: str, force: bool = False) -> int:
        """重建指定类型的索引

        Args:
            index_type: 要重建的索引类型
            force: 是否强制重建（重置所有该类型的索引状态）

        Returns:
            int: 重建的索引数量
        """
        if index_type not in self._indexers:
            logger.error(f"Indexer not available for type: {index_type}")
            return 0

        if force:
            # 强制重建：重置所有该类型的索引状态
            chunks = self._relational_db.list_chunks_metadata()
            reset_count = 0
            for chunk in chunks:
                if hasattr(chunk, 'index_type') and chunk.index_type == index_type:
                    updates = {
                        "status": ChunksStatus.CHUNKED,
                        "index_type": None
                    }
                    if self._relational_db.update_chunks_metadata(chunk.chunks_id, updates):
                        reset_count += 1

            logger.info(f"Reset {reset_count} chunks for index type {index_type}")

        # 重新构建
        unindexed_chunks = self._relational_db.list_chunks_metadata(status=ChunksStatus.CHUNKED)
        relevant_chunks = [chunk for chunk in unindexed_chunks
                          if self._determine_index_type(chunk) == index_type]

        if relevant_chunks:
            return self._build_index_for_type(index_type, relevant_chunks)
        else:
            logger.info(f"No chunks found for rebuilding index type {index_type}")
            return 0

    def get_index_statistics(self) -> Dict[str, Dict[str, int]]:
        """获取索引统计信息

        Returns:
            Dict[str, Dict[str, int]]: 每种索引类型的统计信息
        """
        stats = {}

        for index_type in self._indexers.keys():
            try:
                # 获取该类型的所有chunks
                all_chunks = self._relational_db.list_chunks_metadata()
                type_chunks = [chunk for chunk in all_chunks
                              if hasattr(chunk, 'index_type') and chunk.index_type == index_type]

                stats[index_type] = {
                    'total': len(type_chunks),
                    'indexed': len([c for c in type_chunks if c.status == ChunksStatus.INDEXED]),
                    'pending': len([c for c in type_chunks if c.status == ChunksStatus.CHUNKED]),
                    'failed': len([c for c in type_chunks if c.status == ChunksStatus.FAILED])
                }
            except Exception as e:
                logger.error(f"Failed to get statistics for index type {index_type}: {e}")
                stats[index_type] = {
                    'total': 0,
                    'indexed': 0,
                    'pending': 0,
                    'failed': 0
                }

        return stats

    def get_indexer_health(self) -> Dict[str, Dict[str, Any]]:
        """获取索引器健康状态

        Returns:
            Dict[str, Dict[str, Any]]: 每个索引器的健康状态
        """
        health_status = {}

        for index_type, indexer in self._indexers.items():
            try:
                if hasattr(indexer, 'health_check'):
                    health_status[index_type] = indexer.health_check()
                else:
                    # 基本健康检查
                    health_status[index_type] = {
                        "status": "unknown",
                        "exists": hasattr(indexer, 'index_exists') and indexer.index_exists(),
                        "stats": indexer.get_index_stats() if hasattr(indexer, 'get_index_stats') else {}
                    }
            except Exception as e:
                logger.error(f"Health check failed for indexer {index_type}: {e}")
                health_status[index_type] = {
                    "status": "error",
                    "error": str(e),
                    "exists": False,
                    "stats": {}
                }

        return health_status