import re
from typing import List, Callable, Optional, Tuple
import logging
import jieba
from core.utils.data_model import Document

logger = logging.getLogger(__name__)

class TokenizerManager:
    """
    统一管理分词器的类，支持 jieba、空格分词和自定义分词器。
    支持多进程环境下的初始化和序列化。
    """

    def __init__(self, custom_preprocess_func: Optional[Callable[[str], List[str]]] = None):
        self.custom_preprocess_func = custom_preprocess_func
        self._use_jieba = None  # None 表示未检测，True/False 表示已确定
        self._tokenizer_stats = None
        self._stopwords = ["的", "是", "在", "和", "与", "或", "了", "等", "就", "也",
                          "一", "个", "有", "这", "那", "不", "但", "对", "为", "很"]

    @staticmethod
    def _jieba_tokenize(text: str) -> List[str]:
        """jieba分词（静态方法，便于序列化）"""
        if not text or not text.strip():
            return []
        return list(jieba.cut(text.strip()))

    @staticmethod
    def _whitespace_tokenize(text: str) -> List[str]:
        """空格分词（静态方法，便于序列化）"""
        if not text or not text.strip():
            return []
        return text.strip().split()

    def get_current_tokenizer(self) -> Callable[[str], List[str]]:
        """获取当前使用的分词函数"""
        if self.custom_preprocess_func is not None:
            return self.custom_preprocess_func
        elif self._use_jieba is True:
            return self._jieba_tokenize
        elif self._use_jieba is False:
            return self._whitespace_tokenize
        else:
            # 默认使用空格分词，直到语言检测完成
            return self._whitespace_tokenize

    def detect_language(self, documents: List[Document], sample_size: int = 20,
                       chinese_ratio_threshold: float = 0.1) -> Tuple[bool, dict]:
        """检测文档语言，决定是否使用 jieba"""
        if not documents:
            return False, {"reason": "no_documents"}

        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
        sample_docs = documents[:sample_size] if len(documents) > sample_size else documents

        total_chars, chinese_chars, docs_with_chinese = 0, 0, 0
        for doc in sample_docs:
            content = doc.content or ""
            non_space_content = re.sub(r'\s+', '', content)
            if not non_space_content:
                continue
            doc_total = len(non_space_content)
            doc_chinese = len(chinese_pattern.findall(non_space_content))
            total_chars += doc_total
            chinese_chars += doc_chinese
            if doc_chinese > 0:
                docs_with_chinese += 1

        stats = {
            "total_chars": total_chars,
            "chinese_chars": chinese_chars,
            "chinese_ratio": chinese_chars / max(total_chars, 1),
            "docs_with_chinese": docs_with_chinese,
            "total_sampled_docs": len(sample_docs),
            "docs_with_chinese_ratio": docs_with_chinese / max(len(sample_docs), 1)
        }

        chinese_char_ratio = stats["chinese_ratio"]
        chinese_doc_ratio = stats["docs_with_chinese_ratio"]
        use_jieba = (chinese_char_ratio >= chinese_ratio_threshold or chinese_doc_ratio >= 0.3)

        stats["decision"] = "jieba" if use_jieba else "whitespace"
        stats["reason"] = (
            f"chinese_ratio={chinese_char_ratio:.3f}, "
            f"chinese_doc_ratio={chinese_doc_ratio:.3f}, "
            f"threshold={chinese_ratio_threshold}"
        )
        return use_jieba, stats

    def set_tokenizer_by_detection(self, documents: List[Document]) -> None:
        """根据文档内容自动检测并设置分词器"""
        if self.custom_preprocess_func is not None:
            logger.info("Custom preprocess_func provided, skipping language detection and tokenizer switch.")
            return

        use_jieba, stats = self.detect_language(documents)
        self._tokenizer_stats = stats
        
        if self._use_jieba == use_jieba:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Tokenizer already set to {'jieba' if use_jieba else 'whitespace'}")
            return
            
        self._use_jieba = use_jieba
        logger.info(f"Switched to {'jieba' if use_jieba else 'whitespace'} tokenizer. {stats['reason']}")

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """批量分词"""
        tokenize_func = self.get_current_tokenizer()
        result = []
        for text in texts:
            result.append(tokenize_func(text))
        return result

    def get_tokenizer_info(self) -> str:
        """获取当前分词器信息"""
        if self.custom_preprocess_func is not None:
            return "custom"
        elif self._use_jieba is True:
            return "jieba"
        elif self._use_jieba is False:
            return "whitespace"
        else:
            return "unset"

    def get_stats(self) -> dict:
        """获取分词器统计信息"""
        base_stats = {
            "current_tokenizer": self.get_tokenizer_info(),
            "use_jieba": self._use_jieba,
            "use_custom_preprocess": self.custom_preprocess_func is not None,
            "tokenizers_registered": True  # 假设已注册
        }
        
        if self._tokenizer_stats:
            base_stats.update(self._tokenizer_stats)
            
        return base_stats

    def update_stopwords(self, stopwords: List[str]) -> None:
        """更新停用词列表"""
        self._stopwords = stopwords