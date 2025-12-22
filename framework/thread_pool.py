"""
全局线程池管理器

提供统一的线程池来执行阻塞操作，避免阻塞事件循环。
确保不同功能（用户登录、上传文件、问答、图谱关系）可以并发执行，不会互相阻塞。
支持透传 correlation_id 等 contextvars 到线程池中。
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
    全局线程池管理器
    
    使用单例模式，确保整个应用共享同一个线程池。
    这样可以更好地控制并发数量，避免创建过多线程。
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
            # 根据CPU核心数设置线程池大小
            # 对于I/O密集型任务，可以设置更大的线程数
            # 默认使用 min(32, (os.cpu_count() or 1) + 4)
            import os
            max_workers = min(32, (os.cpu_count() or 1) + 4)
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="rag-arc-worker"
            )
            logger.info(f"Global thread pool initialized with {max_workers} workers")
    
    @property
    def executor(self) -> ThreadPoolExecutor:
        """获取线程池执行器"""
        if self._executor is None:
            self.__init__()
        return self._executor
    
    async def run_blocking(self, func, *args, **kwargs):
        """
        在线程池中运行阻塞函数，自动透传 contextvars（如 correlation_id）
        
        Args:
            func: 要执行的阻塞函数
            *args: 函数的位置参数
            **kwargs: 函数的关键字参数
            
        Returns:
            函数的返回值
        """
        # 捕获当前 context（包含 correlation_id 等）
        ctx = contextvars.copy_context()
        
        # 在线程中运行函数时使用捕获的 context
        def run_with_context():
            return ctx.run(partial(func, *args, **kwargs))
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, run_with_context)
    
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


# 全局单例实例
_global_thread_pool: Optional[GlobalThreadPool] = None


def get_thread_pool() -> GlobalThreadPool:
    """
    获取全局线程池实例
    
    Returns:
        GlobalThreadPool实例
    """
    global _global_thread_pool
    if _global_thread_pool is None:
        _global_thread_pool = GlobalThreadPool()
    return _global_thread_pool


async def run_blocking(func, *args, **kwargs):
    """
    便捷函数：在线程池中运行阻塞函数
    
    Args:
        func: 要执行的阻塞函数
        *args: 函数的位置参数
        **kwargs: 函数的关键字参数
        
    Returns:
        函数的返回值
    """
    return await get_thread_pool().run_blocking(func, *args, **kwargs)
