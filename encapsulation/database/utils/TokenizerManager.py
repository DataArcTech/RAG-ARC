import re
from typing import List, Callable, Optional, Tuple
import logging
import jieba
from pathlib import Path
from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)

class TokenizerManager:
    """
    Unified tokenizer management class supporting jieba, whitespace tokenization and custom tokenizers.
    Supports initialization and serialization in multi-process environments.
    """

    def __init__(self, custom_preprocess_func: Optional[Callable[[str], List[str]]] = None,
                 custom_stopwords_file: Optional[str] = None):
        self.custom_preprocess_func = custom_preprocess_func
        self.custom_stopwords_file = custom_stopwords_file
        self._use_jieba = None  # None means not detected, True/False means determined
        self._tokenizer_stats = None
        self._stopwords = []  # Will be loaded in _load_stopwords

    @staticmethod
    def _jieba_tokenize(text: str) -> List[str]:
        """Jieba tokenization (static method for easy serialization)"""
        if not text or not text.strip():
            return []
        return list(jieba.cut(text.strip()))

    @staticmethod
    def _whitespace_tokenize(text: str) -> List[str]:
        """Whitespace tokenization (static method for easy serialization)"""
        if not text or not text.strip():
            return []
        return text.strip().split()

    def _load_stopwords_from_file(self, file_path: str) -> List[str]:
        """Load stopwords from file
        
        Args:
            file_path: Path to stopwords file
            
        Returns:
            List of stopwords
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"Stopwords file not found: {file_path}, using empty stopwords list")
            return []
        
        stopwords = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        stopwords.append(word)
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    for line in f:
                        word = line.strip()
                        if word and not word.startswith('#'):
                            stopwords.append(word)
            except UnicodeDecodeError:
                logger.error(f"Unable to decode stopwords file: {file_path}")
                return []
        
        logger.info(f"Loaded {len(stopwords)} stopwords from: {file_path}")
        return stopwords

    def _get_default_stopwords_file(self) -> str:
        """Get default stopwords file path based on tokenizer type
        
        Returns:
            Path to default stopwords file
        """
        # Get project root directory
        current_dir = Path(__file__).parent
        stopwords_dir = current_dir / "stopwords"  # utils/stopwords
        
        if self._use_jieba:
            # Use jieba tokenizer, select Chinese stopwords
            default_file = stopwords_dir / "chinese_default.txt"
        else:
            # Use whitespace tokenizer, select English stopwords
            default_file = stopwords_dir / "english_default.txt"
        
        return str(default_file)

    def _load_stopwords(self) -> None:
        """Load stopwords list"""
        if self.custom_stopwords_file:
            # Use custom stopwords file
            self._stopwords = self._load_stopwords_from_file(self.custom_stopwords_file)
        else:
            # Use default stopwords file based on tokenizer type
            if self._use_jieba is not None:  # Tokenizer already determined
                default_file = self._get_default_stopwords_file()
                self._stopwords = self._load_stopwords_from_file(default_file)
            else:
                # Tokenizer not determined yet, use empty list temporarily
                self._stopwords = []

    def get_stopwords(self) -> List[str]:
        """Get current stopwords list
        
        Returns:
            List of stopwords
        """
        if not self._stopwords and self._use_jieba is not None:
            self._load_stopwords()
        return self._stopwords

    def get_current_tokenizer(self) -> Callable[[str], List[str]]:
        """Get currently used tokenizer function"""
        if self.custom_preprocess_func is not None:
            return self.custom_preprocess_func
        elif self._use_jieba is True:
            return self._jieba_tokenize
        elif self._use_jieba is False:
            return self._whitespace_tokenize
        else:
            # Default to whitespace tokenization until language detection is complete
            return self._whitespace_tokenize

    def detect_language(self, chunks: List[Chunk], sample_size: int = 20,
                       chinese_ratio_threshold: float = 0.1) -> Tuple[bool, dict]:
        """Detect chunk language to decide whether to use jieba"""
        if not chunks:
            return False, {"reason": "no_chunks"}

        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
        sample_docs = chunks[:sample_size] if len(chunks) > sample_size else chunks

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

    def set_tokenizer_by_detection(self, chunks: List[Chunk]) -> None:
        """Automatically detect and set tokenizer based on chunk content"""
        if self.custom_preprocess_func is not None:
            logger.info("Custom preprocess_func provided, skipping language detection and tokenizer switch.")
            return

        use_jieba, stats = self.detect_language(chunks)
        self._tokenizer_stats = stats
        
        if self._use_jieba == use_jieba:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Tokenizer already set to {'jieba' if use_jieba else 'whitespace'}")
            return
            
        self._use_jieba = use_jieba
        logger.info(f"Switched to {'jieba' if use_jieba else 'whitespace'} tokenizer. {stats['reason']}")
        
        # Load corresponding stopwords after tokenizer is determined
        self._load_stopwords()

    def batch_tokenize(self, texts: List[str]) -> List[List[str]]:
        """Batch tokenization"""
        tokenize_func = self.get_current_tokenizer()
        result = []
        for text in texts:
            result.append(tokenize_func(text))
        return result

    def get_tokenizer_info(self) -> str:
        """Get current tokenizer information"""
        if self.custom_preprocess_func is not None:
            return "custom"
        elif self._use_jieba is True:
            return "jieba"
        elif self._use_jieba is False:
            return "whitespace"
        else:
            return "unset"

    def get_stats(self) -> dict:
        """Get tokenizer statistics"""
        base_stats = {
            "current_tokenizer": self.get_tokenizer_info(),
            "use_jieba": self._use_jieba,
            "use_custom_preprocess": self.custom_preprocess_func is not None,
            "tokenizers_registered": True  # Assume registered
        }
        
        if self._tokenizer_stats:
            base_stats.update(self._tokenizer_stats)
            
        return base_stats
