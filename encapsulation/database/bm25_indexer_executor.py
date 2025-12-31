import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import jieba

logger = logging.getLogger(__name__)


def init_jieba_worker():
    """Initialize jieba in the worker process to reduce initialization overhead"""
    return jieba


class _BM25IndexBuilderExecutorMixin:
    def _get_executor(self) -> Optional[ProcessPoolExecutor]:
        """Lazy load process pool executor with initializer

        Returns:
            ProcessPoolExecutor instance or None if not available
        """
        max_workers = self.config.max_workers or min(4, multiprocessing.cpu_count() - 1)
        if max_workers > 1 and self._executor is None and not self._executor_closed:
            try:
                self._executor = ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=init_jieba_worker,  # Initialize jieba in each worker process
                )
                logger.debug(f"Process pool executor created with {max_workers} workers")
            except Exception as e:
                logger.error(f"Failed to create process pool executor: {e}")
                self._executor_closed = True
        return self._executor

    def close(self) -> None:
        """Close the process pool executor manually"""
        if self._executor and not self._executor_closed:
            try:
                self._executor.shutdown(wait=True)
                logger.info("Process pool executor closed successfully")
            except Exception as e:
                logger.error(f"Error closing process pool executor: {e}")
            finally:
                self._executor = None
                self._executor_closed = True

    def __del__(self) -> None:
        """Destructor to close process pool"""
        try:
            self.close()
        except Exception as e:
            try:
                logger.error(f"Error in __del__: {e}")
            except Exception:
                pass

