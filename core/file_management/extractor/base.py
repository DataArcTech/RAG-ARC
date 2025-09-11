"""
多次提取实体、关系，避免遗漏。
多轮提取，后续清洗，避免出现错误。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Literal, Callable, TypeVar, Generic
from pydantic import Field
import asyncio
import copy 
import logging

from encapsulation.llm.base import LLMBase, LLMBaseConfig
from core.utils.data_model import Document
from framework.module import AbstractModule
from framework.config import AbstractConfig

# 设置日志记录器
logger = logging.getLogger(__name__)

ConfigType = TypeVar("ConfigType", bound="ExtractorBaseConfig")

class ExtractorBaseConfig(AbstractConfig):
    """
    Abstract base class for all Extractor configurations.
    
    This class defines the common configuration parameters for all extractors.
    Subclasses must:
    1. Define a unique `type` field with a Literal value
    2. Implement the `build()` method to create the corresponding Extractor instance
    
    Attributes:
        type (Literal): The type of the extractor, must be overridden by subclasses
        max_concurrent (int): Maximum number of concurrent operations (default: 100)
        enable_cleaning (bool): Whether to enable cleaning functionality (default: True)
        max_rounds (int): Maximum number of extraction rounds (default: 3)
        llm_config (LLMBaseConfig): Configuration for the LLM to be used
    """
    type: Literal["base_extractor"] = "base_extractor"

    max_concurrent: int = Field(default=100, description="Maximum number of concurrent operations", ge=1)
    enable_cleaning: bool = Field(default=True, description="Whether to enable cleaning functionality")
    max_rounds: int = Field(default=3, description="Maximum number of extraction rounds", ge=1)

    llm_config: LLMBaseConfig = Field(default=None, description="Configuration for the LLM to be used")

    @abstractmethod
    def build(self) -> "ExtractorBase":
        """
        Build the Extractor instance.
        
        This method must be implemented by subclasses to create and return 
        the corresponding Extractor instance.
        
        Returns:
            ExtractorBase: The constructed extractor instance
        """
        raise NotImplementedError("Subclasses must implement build() method")


class ExtractorBase(AbstractModule, Generic[ConfigType]):
    """
    提取器基类，定义了所有提取器的通用接口和功能。
    
    核心功能：
    - 多轮提取
    - 可选的提取清洗
    - 并发控制和批量处理
    - 同步/异步调用接口
    - 进度显示支持
    
    子类需要实现：
    - _aextract: 异步提取单个文档的图结构
    - _aclean: 异步清洗单个文档的图结构（可选）
    """

    config: ConfigType
    




    # ==================== 抽象与可重写方法 ====================
    @abstractmethod
    async def _aextract(
        self, 
        document: Document,  
        history: Dict[str, List]
    ) -> Document:
        """
        异步单次抽取文档的图结构（抽象方法，子类必须实现）
        
       
        Args:
            document: 待处理的文档
            history: 抽取历史，包含已提取的实体和关系
            
        Returns:
            Document: 处理后的文档，metadata中包含【本轮】新提取的图结构
        """
        pass

    async def _aclean(
        self, 
        document: Document
    ) -> Document:
        """
        异步清洗单个文档的图结构（可选实现，默认不做处理）
        
        子类可以重写此方法来实现自定义的清洗逻辑
        
        Args:
            document: 待清洗的文档（已包含提取的图结构）
            
        Returns:
            Document: 清洗后的文档
        """
        # 判断是否被子类重写
        if inspect.getattr_static(self.__class__, "_aclean") is ExtractorBase._aclean:
            if getattr(self.config, "enable_cleaning", False):
                logger.warning(
                    f"{self.__class__.__name__}.enable_cleaning=True 但 _aclean 方法未实现，清洗操作不会生效。"
                )

        return document

    # ==================== 核心处理逻辑 ====================

    async def _run_with_queue(
        self,
        documents: List[Document],
        worker_fn: Callable,
        show_progress: bool = False,
        progress_desc: str = "Processing documents",
        use_index: bool = False
    ) -> List[Document]:
        """
        通用的队列消费者执行器，用于处理文档列表。
        
        Args:
            documents: 待处理的文档列表。
            worker_fn: 处理单个文档的函数。
            show_progress: 是否显示进度条。
            progress_desc: 进度条描述。
            use_index: 是否在队列中使用索引以保持结果顺序。
            
        Returns:
            处理完成的文档列表。
        """
        results = [None] * len(documents) if use_index else []
        queue_size = max(self.config.max_concurrent *2, len(documents)//10)
        queue = asyncio.Queue(maxsize=queue_size)
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        sentinel = object()

        # 进度条 (可选)
        pbar = None
        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio
                pbar = tqdm_asyncio(total=len(documents), desc=progress_desc)
            except ImportError:
                logger.warning("tqdm not installed, progress disabled.")
                show_progress = False

        async def consumer():
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                idx, doc = item if use_index else (None, item)
                try:
                    async with semaphore:
                        result = await worker_fn(doc)
                    if use_index:
                        results[idx] = result
                    else:
                        results.append(result)
                except Exception as e:
                    logger.error(f"处理文档时发生错误: {e}", exc_info=True)
                    # 在使用索引时，确保即使出错也要设置结果以避免None值
                    if use_index:
                        results[idx] = doc  # 插入原始文档
                    else:
                        results.append(doc)  # 插入原始文档
                finally:
                    if pbar:
                        pbar.update(1)
                    queue.task_done()

        consumers_num = min(self.config.max_concurrent, len(documents))
        consumers = [asyncio.create_task(consumer()) for _ in range(consumers_num)]

        async def producer():
            for idx, doc in enumerate(documents) if use_index else documents:
                await queue.put((idx, doc) if use_index else doc)
            # 为每个 consumer 放入一个 sentinel
            for _ in range(consumers_num):
                await queue.put(sentinel)

        await asyncio.create_task(producer())
        await queue.join()
        consumer_results = await asyncio.gather(*consumers, return_exceptions=True)
        # 检查consumer任务中的异常
        for i, res in enumerate(consumer_results):
            if isinstance(res, Exception):
                logger.error(f"Consumer task {i} failed with exception: {res}", exc_info=True)
        if pbar:
            pbar.close()
        return results

    def _merge_graph_data(
        self,
        history: Dict[str, List],
        new_extraction: Document
    ) -> Dict[str, List]:
        """
        合并历史记录和新提取结果，自动去重。
        
        子类可以重写此方法来实现自定义的合并逻辑（例如，处理属性冲突、基于置信度合并等）。
        """
        new_metadata = new_extraction.metadata or {}

        # 使用集合来提高去重效率，避免在大规模图谱提取任务中成为性能瓶颈
        # 先创建历史实体和关系的ID集合
        existing_entity_ids = {e.get('id') for e in history.get('entities', [])}
        existing_relation_ids = {r['id'] for r in history.get('relations', [])}
        
        # 合并实体数据
        entities = list(history.get('entities', []))
        for entity in new_metadata.get('entities', []):
            if entity['id'] not in existing_entity_ids:
                entities.append(entity)
                existing_entity_ids.add(entity['id'])
        
        # 合并关系数据
        relations = list(history.get('relations', []))
        for relation in new_metadata.get('relations', []):
            if relation['id'] not in existing_relation_ids:
                relations.append(relation)
                existing_relation_ids.add(relation['id'])

        return {
            'entities': entities,
            'relations': relations
        }


    async def _process_single_document(
        self,
        document: Document,
        should_clean: bool
    ) -> Document:
        """完整处理单个文档：多轮提取 + 可选清洗。"""
        # 检查max_rounds参数
        if self.config.max_rounds <= 0:
            logger.warning("max_rounds should be greater than 0. Returning original document.")
            return document
            
        metadata_copy = {
            'entities': list(document.metadata.get('entities', [])),
            'relations': list(document.metadata.get('relations', [])),
            # 其他字段如需保留，可浅拷贝
            **{k: v for k, v in (document.metadata or {}).items() if k not in {'entities', 'relations'}}
        }
        doc_copy = Document(id=document.id, content=document.content, metadata=metadata_copy)

        # 初始化历史记录
        history = {
            'entities': list(doc_copy.metadata.get('entities', [])),
            'relations': list(doc_copy.metadata.get('relations', [])),
        }

        # 多轮提取
        for i in range(self.config.max_rounds):
            logger.debug(f"开始第 {i + 1}/{self.config.max_rounds} 轮提取...")
            
            extracted_doc = await self._aextract(doc_copy, history)
            
            new_entities = extracted_doc.metadata.get('entities', [])
            new_relations = extracted_doc.metadata.get('relations', [])

            # 如果没有新内容，则提前终止
            if not new_entities and not new_relations:
                logger.info(f"第 {i + 1} 轮未发现新内容，提前结束。")
                break
            
            # 合并新结果到历史记录，并更新文档元数据以备下一轮使用
            history = self._merge_graph_data(history, extracted_doc)
            doc_copy.metadata.update(history)

        # 清洗（如果启用）
        if should_clean and self.config.enable_cleaning:
            logger.info("开始清洗图结构...")
            return await self._aclean(doc_copy)
        
        return doc_copy

    # ==================== 公共调用接口 ====================

    async def acall(
        self,
        documents: List[Document],
        show_progress: bool = False,
        enable_cleaning: Optional[bool] = None
    ) -> List[Document]:
        """
        异步处理文档列表（提取 + 清洗），采用生产者-消费者模式优化资源使用。

        Args:
            documents: 待处理的文档列表。
            show_progress: 是否显示进度条。
            enable_cleaning: 是否启用清洗，会覆盖实例的默认设置。

        Returns:
            处理完成的文档列表。
        """
        if not documents:
            logger.info("文档列表为空，无需处理。")
            return []

        # 确定本次调用是否需要清洗，避免修改实例状态
        should_clean = self.config.enable_cleaning if enable_cleaning is None else enable_cleaning
        
        # 定义处理单个文档的包装函数
        async def process_document(document):
            return await self._process_single_document(document, should_clean)
        
        # 使用通用队列执行器处理文档
        results = await self._run_with_queue(
            documents, 
            process_document, 
            show_progress, 
            "处理文档"
        )
        
        action = "提取并清洗" if should_clean else "提取"
        if show_progress:
            logger.info(f"图结构{action}完成。")
            
        return results

    async def _run_pipeline(
        self,
        documents: List[Document],
        process_func: Callable,
        show_progress: bool = False,
        progress_desc: str = "Processing documents",
        process_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        通用的管道执行器，用于处理文档列表。
        
        Args:
            documents: 待处理的文档列表。
            process_func: 处理单个文档的函数。
            show_progress: 是否显示进度条。
            progress_desc: 进度条描述。
            process_kwargs: 传递给处理函数的额外参数。
            
        Returns:
            处理完成的文档列表。
        """
        if not documents:
            logger.info("文档列表为空，无需处理。")
            return []

        process_kwargs = process_kwargs or {}

        # 定义处理单个文档的包装函数
        async def process_document(document):
            return await process_func(document, **process_kwargs)
        
        # 使用通用队列执行器处理文档
        results = await self._run_with_queue(
            documents, 
            process_document, 
            show_progress, 
            progress_desc
        )
        
        if show_progress:
            logger.info(f"{progress_desc}完成。")
            
        return results
    


    def __call__(
        self, 
        documents: List[Document], 
        show_progress: bool = False,
        enable_cleaning: Optional[bool] = None
    ) -> List[Document]:
        """同步接口：提取和清洗图结构"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.acall(documents, show_progress=show_progress, enable_cleaning=enable_cleaning)
            )
        else:
            return loop.run_until_complete(
                self.acall(documents, show_progress=show_progress, enable_cleaning=enable_cleaning)
            )

    # ==================== 专用接口 ====================

    async def aextract_only(
        self, 
        documents: List[Document], 
        show_progress: bool = False
    ) -> List[Document]:
        """仅执行提取操作，跳过清洗"""
        return await self._run_pipeline(
            documents=documents,
            process_func=self._process_single_document,
            process_kwargs={'should_clean': False},
            show_progress=show_progress,
            progress_desc="提取文档"
        )

    def extract_only(
        self, 
        documents: List[Document], 
        show_progress: bool = False
    ) -> List[Document]:
        """同步接口：仅执行提取操作，跳过清洗"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aextract_only(documents, show_progress=show_progress))
        else:
            return loop.run_until_complete(self.aextract_only(documents, show_progress=show_progress))

    async def aclean_only(
        self, 
        documents: List[Document], 
        show_progress: bool = False
    ) -> List[Document]:
        """
        异步清洗文档列表，采用生产者-消费者模式优化资源使用。

        Args:
            documents: 待清洗的文档列表。
            show_progress: 是否显示进度条。

        Returns:
            清洗完成的文档列表。
        """
        return await self._run_pipeline(
            documents=documents,
            process_func=self._aclean,
            show_progress=show_progress,
            progress_desc="清洗文档"
        )

    def clean_only(
        self, 
        documents: List[Document], 
        show_progress: bool = False
    ) -> List[Document]:
        """同步接口：仅执行清洗操作"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aclean_only(documents, show_progress=show_progress))
        else:
            return loop.run_until_complete(self.aclean_only(documents, show_progress=show_progress))

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__