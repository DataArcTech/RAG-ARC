from application.rag_inference.graph_store_provider import get_graph_store
from framework.register import Register


class _DummyRag:
    def __init__(self, store):
        self._store = store

    def get_graph_store(self):
        return self._store


def test_graph_store_provider_reflects_latest_registration():
    reg = Register()
    previous = reg.registrations.get("rag_inference")
    try:
        store1 = object()
        store2 = object()

        reg.registrations["rag_inference"] = _DummyRag(store1)
        assert get_graph_store() is store1

        reg.registrations["rag_inference"] = _DummyRag(store2)
        assert get_graph_store() is store2
    finally:
        if previous is None:
            reg.registrations.pop("rag_inference", None)
        else:
            reg.registrations["rag_inference"] = previous
