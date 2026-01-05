import pytest

from api.retrieval_api import RetrievalAPI


class _DummyRetriever:
    def __init__(self):
        self.calls = []

    def invoke(self, query, k=5, owner_id=None, **kwargs):
        self.calls.append({'query': query, 'owner_id': owner_id, 'k': k})
        return []


def test_search_requires_owner_id():
    api = RetrievalAPI()
    api.retrievers['demo'] = _DummyRetriever()

    with pytest.raises(ValueError, match="owner_id is required"):
        api.search('demo', 'hello world')


def test_search_passes_owner_id_to_retriever():
    api = RetrievalAPI()
    dummy = _DummyRetriever()
    api.retrievers['demo'] = dummy

    api.search('demo', 'hello world', owner_id='owner-123', k=2)
    assert dummy.calls == [{'query': 'hello world', 'owner_id': 'owner-123', 'k': 2}]
