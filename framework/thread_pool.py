"""
Global thread-pool manager.

Provides a shared thread pool for running blocking work without blocking the event loop.
Ensures different workflows (login, file upload, Q&A, graph operations) can run concurrently.
Propagates contextvars (e.g., correlation_id) into worker threads.
"""
import asyncio
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

logger = logging.getLogger(__name__)


class GlobalThreadPool:
    """
    Global thread-pool manager.
    
    Uses a singleton pattern so the whole application shares one pool.
    This helps control concurrency and avoids creating too many threads.
    """
    _instance: Optional['GlobalThreadPool'] = None
    _executor: Optional[ThreadPoolExecutor] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._executor is None:
            # Size the pool based on CPU cores.
            # For I/O-bound workloads, a larger pool can be reasonable.
            # Default: min(32, (os.cpu_count() or 1) + 4)
            import os
            max_workers = min(32, (os.cpu_count() or 1) + 4)
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="rag-arc-worker"
            )
            logger.info(f"Global thread pool initialized with {max_workers} workers")
    
    @property
    def executor(self) -> ThreadPoolExecutor:
        """Return the thread pool executor."""
        if self._executor is None:
            self.__init__()
        return self._executor
    
    async def run_blocking(self, func, *args, **kwargs):
        """
        Run a blocking function in the pool, propagating contextvars (e.g., correlation_id).
        
        Args:
            func: blocking callable
            *args: positional args
            **kwargs: keyword args
            
        Returns:
            The callable result.
        """
        # Capture the current context (including correlation_id, etc.).
        ctx = contextvars.copy_context()
        
        # Run with captured context inside the worker thread.
        def run_with_context():
            return ctx.run(partial(func, *args, **kwargs))
        
        # NOTE:
        # We intentionally avoid asyncio's run_in_executor here.
        # In some constrained runtimes we observed missed wakeups for the 2nd+ executor future,
        # causing await to hang forever. Polling the concurrent.futures.Future is more robust.
        future = self.executor.submit(run_with_context)

        sleep_seconds = 0.0
        try:
            while not future.done():
                await asyncio.sleep(sleep_seconds)
                if sleep_seconds == 0.0:
                    sleep_seconds = 0.001
                elif sleep_seconds < 0.05:
                    sleep_seconds = min(0.05, sleep_seconds * 2)
            return future.result()
        except asyncio.CancelledError:
            future.cancel()
            raise
    
    def shutdown(self, wait: bool = True):
        """
        关闭线程池
        
        Args:
            wait: 是否等待所有任务完成
        """
        if self._executor is not None:
            logger.info("Shutting down global thread pool...")
            self._executor.shutdown(wait=wait)
            self._executor = None
            logger.info("Global thread pool shut down")


# Global singleton instance.
_global_thread_pool: Optional[GlobalThreadPool] = None


def get_thread_pool() -> GlobalThreadPool:
    """
    Get the global thread pool instance.
    
    Returns:
        GlobalThreadPool instance.
    """
    global _global_thread_pool
    if _global_thread_pool is None:
        _global_thread_pool = GlobalThreadPool()
    return _global_thread_pool


async def run_blocking(func, *args, **kwargs):
    """
    Convenience helper: run a blocking function in the thread pool.
    
    Args:
        func: blocking callable
        *args: positional args
        **kwargs: keyword args
        
    Returns:
        The callable result.
    """
    return await get_thread_pool().run_blocking(func, *args, **kwargs)
