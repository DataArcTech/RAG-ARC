"""Shared thread-pool utilities for running blocking work safely."""
import asyncio
import atexit
import contextvars
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

_EXECUTOR: ThreadPoolExecutor | None = None


def get_shared_thread_pool(*, max_workers: int | None = None) -> ThreadPoolExecutor:
    """Return a process-wide shared ThreadPoolExecutor.

    The pool size can be configured via `RAGARC_SHARED_THREADPOOL_MAX_WORKERS`.
    """

    global _EXECUTOR
    if _EXECUTOR is not None:
        return _EXECUTOR

    env = os.getenv("RAGARC_SHARED_THREADPOOL_MAX_WORKERS")
    if max_workers is None and env:
        try:
            max_workers = int(env)
        except ValueError:
            max_workers = None

    _EXECUTOR = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ragarc")
    atexit.register(_shutdown_executor)
    return _EXECUTOR


def _shutdown_executor() -> None:
    global _EXECUTOR
    executor = _EXECUTOR
    _EXECUTOR = None
    if executor is None:
        return
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        executor.shutdown(wait=False)


async def run_blocking(func: Callable[..., T], *args: Any, executor: Optional[ThreadPoolExecutor] = None, **kwargs: Any) -> T:
    """Run a blocking callable in the shared thread pool, preserving contextvars (e.g., correlation_id)."""
    
    # 捕获当前 context（包含 correlation_id 等）
    ctx = contextvars.copy_context()
    
    # 在线程中运行函数时使用捕获的 context
    def run_with_context():
        return ctx.run(partial(func, *args, **kwargs))
    
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor or get_shared_thread_pool(), run_with_context)


async def run_coroutine_in_thread(coro_func: Callable[..., Any], *args: Any, executor: Optional[ThreadPoolExecutor] = None, **kwargs: Any) -> Any:
    """Run an async callable in a thread, returning its result."""

    def _runner() -> Any:
        return asyncio.run(coro_func(*args, **kwargs))

    return await run_blocking(_runner, executor=executor or get_shared_thread_pool())

