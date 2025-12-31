import logging
from typing import List

from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)


class _BM25IndexBuilderTokenizationMixin:
    def _set_tokenizer(self, chunks: List[Chunk]):
        """Set tokenizer (proxied to TokenizerManager)"""
        self.tokenizer_manager.set_tokenizer_by_detection(chunks)

    def _set_tokenizer_from_existing_index(self):
        """Set tokenizer based on existing index"""
        try:
            if self._index is None:
                logger.warning("Index not loaded, cannot set tokenizer from existing index")
                return

            # Check if index has chunks
            searcher = self._index.searcher()
            num_docs = searcher.num_docs

            if num_docs == 0:
                logger.warning("Index is empty, using default whitespace tokenizer")
                return

            # For existing Chinese indices, set tokenizer to jieba
            logger.info(f"Loading existing index with {num_docs} chunks, setting tokenizer to jieba for Chinese content")
            self.tokenizer_manager._use_jieba = True
            self.tokenizer_manager._load_stopwords()

        except Exception as e:
            logger.warning(f"Failed to set tokenizer from existing index: {e}, using default whitespace tokenizer")

    def _tokenize_batch_sequential(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts sequentially (single process)

        Args:
            texts: List of texts to tokenize

        Returns:
            List of tokenized texts
        """
        return self.tokenizer_manager.batch_tokenize(texts)

    def _tokenize_batch_parallel(self, texts: List[str]) -> List[List[str]]:
        """Tokenize texts in parallel (multiprocessing)

        Args:
            texts: List of texts to tokenize

        Returns:
            List of tokenized texts
        """
        executor = self._get_executor()
        if not executor or len(texts) <= self.config.tokenize_batch_size:
            return self._tokenize_batch_sequential(texts)

        # Split texts into batches
        batches = [
            texts[i : i + self.config.tokenize_batch_size]
            for i in range(0, len(texts), self.config.tokenize_batch_size)
        ]
        results = []

        # Create serializable tokenization tasks
        futures = []
        for batch in batches:
            # Since TokenizerManager may contain non-serializable custom functions,
            # we directly use the current instance's tokenization method
            future = executor.submit(self._tokenize_batch_sequential, batch)
            futures.append(future)

        # Collect results
        for future in futures:
            try:
                results.extend(future.result(timeout=60))
            except Exception as e:
                logger.warning(f"Parallel tokenization failed, fallback to sequential: {e}")
                return self._tokenize_batch_sequential(texts)
        return results

