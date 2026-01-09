from types import SimpleNamespace

import numpy as np

from core.retrieval.graph_retrieveal.pruned_hipporag_facts import _PrunedHippoRAGFactsMixin


class _FakeChunk:
    def __init__(self, owner_id: str):
        self.owner_id = owner_id
        self.metadata = {"owner_id": owner_id}


class _FakeFaissIndex:
    def __init__(self, total: int, max_return: int):
        self.ntotal = total
        self._max_return = max_return
        self.calls: list[int] = []

    def search(self, query_vector, k):  # noqa: ANN001
        self.calls.append(int(k))
        scores = np.zeros((1, k), dtype=np.float32)
        indices = -np.ones((1, k), dtype=np.int64)
        n = min(int(k), int(self._max_return))
        if n > 0:
            indices[0, :n] = np.arange(n, dtype=np.int64)
            scores[0, :n] = np.linspace(1.0, 0.0, n, dtype=np.float32)
        return scores, indices


class _Runner(_PrunedHippoRAGFactsMixin):
    def __init__(self):
        self.config = SimpleNamespace(fact_retrieval_top_k=2)

        docstore = {}
        index_to_docstore_id = {}
        # First 50 hits belong to other owner; our owner's facts are later.
        for i in range(80):
            fid = f"f{i}"
            owner = "owner-b" if i < 50 else "owner-a"
            docstore[fid] = _FakeChunk(owner_id=owner)
            index_to_docstore_id[i] = fid

        self.graph_store = SimpleNamespace(
            fact_faiss_db=SimpleNamespace(
                index=_FakeFaissIndex(total=1000, max_return=80),
                config=SimpleNamespace(metric="ip", normalize_L2=False),
                index_to_docstore_id=index_to_docstore_id,
                deleted_ids=set(),
                docstore=docstore,
            )
        )

        self.embedding_model = SimpleNamespace(embed=lambda x: np.zeros(3, dtype=np.float32))

    @staticmethod
    def _owner_to_str(owner_id):  # noqa: ANN001
        return str(owner_id) if owner_id is not None else ""


def test_fact_owner_filter_adapts_k_until_hits_found() -> None:
    runner = _Runner()
    scores, fact_ids = runner._get_fact_scores_faiss("q", owner_id="owner-a")

    assert len(fact_ids) >= 2
    assert all(runner.graph_store.fact_faiss_db.docstore[fid].owner_id == "owner-a" for fid in fact_ids)
    assert runner.graph_store.fact_faiss_db.index.calls[0] == 20  # 2*10
    assert runner.graph_store.fact_faiss_db.index.calls[-1] >= 80


class _RunnerNeedsMoreHits(_PrunedHippoRAGFactsMixin):
    def __init__(self):
        self.config = SimpleNamespace(fact_retrieval_top_k=20)

        docstore = {}
        index_to_docstore_id = {}
        # For k=200, only 5 hits belong to owner-a (non-empty but insufficient).
        # Owner-a facts appear heavily only after rank 200.
        for i in range(500):
            fid = f"f{i}"
            if i < 200:
                owner = "owner-a" if i in {10, 50, 90, 130, 170} else "owner-b"
            else:
                owner = "owner-a"
            docstore[fid] = _FakeChunk(owner_id=owner)
            index_to_docstore_id[i] = fid

        self.graph_store = SimpleNamespace(
            fact_faiss_db=SimpleNamespace(
                index=_FakeFaissIndex(total=2000, max_return=500),
                config=SimpleNamespace(metric="ip", normalize_L2=False),
                index_to_docstore_id=index_to_docstore_id,
                deleted_ids=set(),
                docstore=docstore,
            )
        )

        self.embedding_model = SimpleNamespace(embed=lambda x: np.zeros(3, dtype=np.float32))

    @staticmethod
    def _owner_to_str(owner_id):  # noqa: ANN001
        return str(owner_id) if owner_id is not None else ""


def test_fact_owner_filter_keeps_searching_until_desired_hits() -> None:
    runner = _RunnerNeedsMoreHits()
    scores, fact_ids = runner._get_fact_scores_faiss("q", owner_id="owner-a")

    assert len(fact_ids) >= 20
    assert all(runner.graph_store.fact_faiss_db.docstore[fid].owner_id == "owner-a" for fid in fact_ids)
    assert runner.graph_store.fact_faiss_db.index.calls[0] == 200  # 20*10
    assert runner.graph_store.fact_faiss_db.index.calls[-1] > 200
