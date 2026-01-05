import asyncio

from application.knowledge.module import Knowledge


class _StubTaskQueue:
    def update_task_run(self, *args, **kwargs):  # noqa: ANN001, ARG002
        return None

    def append_progress_event(self, *args, **kwargs):  # noqa: ANN001, ARG002
        return None


class _StubIndex:
    def __init__(self) -> None:
        self.delete_calls = 0

    def delete_file_data(self, file_id: str, **kwargs):  # noqa: ARG002
        self.delete_calls += 1
        return {"success": True}

    async def index_file(self, file_id: str):  # noqa: ARG002
        await asyncio.sleep(0)
        return {"success": True, "file_id": file_id}


async def test_knowledge_indexing_cleans_before_each_run():
    knowledge = Knowledge.__new__(Knowledge)
    knowledge.indexing_semaphore = asyncio.Semaphore(1)  # type: ignore[attr-defined]
    knowledge.task_queue = _StubTaskQueue()  # type: ignore[attr-defined]
    knowledge.file_index = _StubIndex()  # type: ignore[attr-defined]

    knowledge._is_file_marked_for_deletion = lambda doc_id: False  # type: ignore[attr-defined]
    knowledge._run_coroutine_in_thread = lambda coro_func, *a, **k: coro_func(*a, **k)  # type: ignore[attr-defined]

    async def _run_blocking(func, *a, **k):  # noqa: ANN001
        return await asyncio.to_thread(func, *a, **k)

    knowledge._run_blocking = _run_blocking  # type: ignore[attr-defined]

    await knowledge._index_file_background("file_1", task_run_id="run_1")  # type: ignore[attr-defined]
    await knowledge._index_file_background("file_1", task_run_id="run_2")  # type: ignore[attr-defined]

    assert knowledge.file_index.delete_calls == 2  # type: ignore[attr-defined]

