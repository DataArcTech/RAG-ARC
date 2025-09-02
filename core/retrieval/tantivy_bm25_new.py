import logging
from typing import Any, List, Callable, Optional, Dict, Union, Tuple
from tantivy import Index, Query, Occur, Order

from core.utils.data_model import Document

logger = logging.getLogger(__name__)


class TantivyBM25Retriever:
    """
    基于 Tantivy 索引的高性能 BM25 检索器

    改进特性：
    - 支持任意元数据字段过滤（动态字段名）# NOTE 暂时只支持"source"、"tag"，任意字段待Index部分实现
    - 查询词为空时返回空列表（不报错）
    - 支持短语查询（phrase_query）提升相关性
    - 更健壮的异常处理与日志
    - 支持排序字段与自定义排序方向
    """

    def __init__(
        self,
        index: Index,
        preprocess_func: Callable[[str], List[str]],
        top_k: int = 10,
        stopwords: Optional[List[str]] = None,
        use_phrase_query: bool = False,  # 是否启用短语查询
        **kwargs: Any
    ):
        if not isinstance(index, Index):
            raise TypeError(f"期望 tantivy.Index 类型，但得到 {type(index)}")

        self.index = index
        self.preprocess_func = preprocess_func
        self.top_k = top_k
        self.stopwords = set(stopwords or [])
        self.use_phrase_query = use_phrase_query
        self.searcher = self.index.searcher()

    def reload_searcher(self):
        """重新加载搜索器以反映最新索引状态"""
        try:
            self.searcher = self.index.searcher()
            logger.debug("Searcher reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading searcher: {e}")
            raise

    def _build_filter_query(self, filters: Dict[str, Union[str, List[str]]]) -> List[Tuple[Occur, Query]]:
        """构建动态过滤查询，支持任意字段"""
        filter_queries = []
        for field_name, values in filters.items():
            if not isinstance(values, list):
                values = [values]
            if not values:
                continue
            try:
                q = Query.term_set_query(self.index.schema, field_name, values)
                filter_queries.append((Occur.Must, q))
            except Exception as e:
                logger.warning(f"Skipping invalid filter field '{field_name}': {e}")
        return filter_queries

    def _build_main_query(self, query_tokens: List[str]) -> Query:
        """构建主查询，支持普通 BM25 或短语查询"""
        if not query_tokens:
            return Query.all_query()

        # 移除停用词（可选）
        filtered_tokens = [t for t in query_tokens if t not in self.stopwords]
        if not filtered_tokens:
            return Query.all_query()

        if self.use_phrase_query and len(filtered_tokens) > 1:
            # 使用短语查询（词序敏感，更精确）
            try:
                phrase_q = Query.phrase_query(self.index.schema, "content_tokens", filtered_tokens)
                return phrase_q
            except Exception as e:
                logger.warning(f"Falling back to term query due to phrase query error: {e}")

        # 默认：BM25 多词查询
        query_str = " ".join(filtered_tokens)
        fields = ["content_tokens"]
        return self.index.parse_query(query_str, fields)

    def _get_relevant_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        order_by_field: Optional[str] = None,
        order_desc: bool = True,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        执行检索并返回结构化结果

        Returns:
            List[Dict] 包含：
            - id, content, metadata
            - score (float)
        """
        top_k = top_k or self.top_k
        filters = filters or {}

        if not query.strip():
            logger.info("Empty query received, returning empty results.")
            return []

        # 1. 分词预处理
        try:
            query_tokens = self.preprocess_func(query)
            logger.debug(f"Query tokens: {query_tokens}")
        except Exception as e:
            logger.error(f"Error during query preprocessing: {e}")
            return []

        # 2. 构建主查询 + 过滤器
        main_query = self._build_main_query(query_tokens)
        filter_subqueries = self._build_filter_query(filters)

        final_query = (
            Query.boolean_query([(Occur.Must, main_query)] + filter_subqueries)
            if filter_subqueries else main_query
        )

        # 3. 计算实际搜索 top_k（过滤模式下扩大搜索范围）
        search_top_k = top_k * 3 if filter_subqueries else top_k

        # 4. 执行搜索
        try:
            order = Order.Desc if order_desc else Order.Asc
            search_result = self.searcher.search(
                final_query,
                limit=search_top_k,
                order_by_field=order_by_field,
                order=order
            )
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []

        # 5. 组装结果
        results = []
        for score, doc_address in search_result.hits[:top_k]:  # 截断到 top_k
            try:
                tantivy_doc = self.searcher.doc(doc_address)
                doc_data = {
                    "id": tantivy_doc.get_first("id") or "",
                    "content": tantivy_doc.get_first("content") or "",
                    "metadata": tantivy_doc.get_first("metadata") or {},
                    "score": float(score),
                }

                results.append(doc_data)
            except Exception as e:
                logger.warning(f"Failed to parse document from index: {e}")
                continue

        logger.info(f"Retrieved {len(results)} documents for query: '{query}'")
        return results

    def invoke(self, query: str, **kwargs) -> List[Document]:
        """
        返回 Document 对象列表
        """
        raw_results = self._get_relevant_documents(query, **kwargs)
        return [
            Document(
                id=res["id"],
                content=res["content"],
                metadata={**res.get("metadata", {}), "score": res["score"]}
            )
            for res in raw_results
        ]

    def get_relevant_documents_with_score(self, query: str, **kwargs) -> List[Tuple[Document, float]]:
        """返回文档和原始分数的元组列表"""
        raw_results = self._get_relevant_documents(query, **kwargs)
        return [
            (
                Document(id=res["id"], content=res["content"], metadata=res.get("metadata", {})),
                res["score"]
            )
            for res in raw_results
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(top_k={self.top_k}, "
            f"use_phrase={self.use_phrase_query})"
        )