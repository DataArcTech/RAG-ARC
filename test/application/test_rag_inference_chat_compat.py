import uuid

from application.rag_inference.module import RAGInference


def test_chat_keeps_backward_compatible_return_shape() -> None:
    """
    chat_async may return extra experimental fields, but chat() must remain 4-tuple
    for backward compatibility (answer, chunks, subgraph_data, subgraph_info).
    """

    async def _chat_async_stub(*_args, **_kwargs):  # noqa: ANN001
        return ("answer", [], None, None, {"raw": True}, {"mindmap": True})

    rag = RAGInference.__new__(RAGInference)
    # Bind the coroutine method to the instance.
    rag.chat_async = _chat_async_stub.__get__(rag, RAGInference)  # type: ignore[method-assign]

    out = rag.chat("q", uuid.uuid4())
    assert isinstance(out, tuple)
    assert len(out) == 4
    assert out[0] == "answer"

