from typing import Literal, Optional, Dict, Any
from pydantic import Field, field_validator
from framework.config import AbstractConfig
from encapsulation.database.bm25_indexer import BM25IndexBuilder


class BM25IndexBuilderConfig(AbstractConfig):
    """BM25索引构建器配置"""
    type: Literal["bm25_indexer"] = "bm25_indexer"

    # 核心配置
    index_path: str = Field(description="索引存储路径")
    bm25_k1: float = Field(default=1.2, description="BM25 k1参数")
    bm25_b: float = Field(default=0.75, description="BM25 b参数")

    # 可选配置
    preprocess_func_name: Optional[str] = Field(default=None, description="预处理函数名")
    stopwords_file: Optional[str] = Field(default=None, description="停用词文件路径")
    writer_heap_size: Optional[int] = Field(default=None, description="写入器堆大小")
    batch_size: int = Field(default=50, description="批处理大小")
    tokenize_batch_size: int = Field(default=200, description="分词批处理大小")
    max_workers: Optional[int] = Field(default=None, description="最大工作进程数")
    progress_interval: int = Field(default=500, description="进度报告间隔")
    enable_gc: bool = Field(default=True, description="是否启用垃圾回收")
    queue_maxsize: int = Field(default=1000, description="队列最大大小")

    # 搜索配置
    search_kwargs: Dict[str, Any] = Field(
        default_factory=lambda: {
            "use_phrase_query": False,
            "k": 5,
            "with_score": True
        },
        description="搜索参数配置"
    )
    k: int = Field(default=5, description="默认返回结果数量")
    with_score: bool = Field(default=True, description="是否返回分数")

    
    @field_validator("bm25_k1")
    @classmethod
    def validate_bm25_k1(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"bm25_k1 must be greater than 0, but got {v}")
        return v
    
    @field_validator("bm25_b")
    @classmethod
    def validate_bm25_b(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError(f"bm25_b must be between 0 and 1, but got {v}")
        return v

    def build(self):
        return BM25IndexBuilder(config=self)
