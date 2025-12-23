import os
import sys
from dataclasses import dataclass

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from core.utils.retrieval_helper import RetrievalHelper
from encapsulation.data_model.schema import Chunk


class _StubFaissIndex:
    def __init__(self):
        self.ntotal = 1

    def search(self, _query_vector: np.ndarray, _k: int):
        distances = np.array([[0.99]], dtype=np.float32)
        indices = np.array([[0]], dtype=np.int64)
        return distances, indices


@dataclass
class _StubConfig:
    normalize_L2: bool = False
    metric: str = "cosine"


class _StubFaissVectorDB:
    def __init__(self):
        self.index = _StubFaissIndex()
        self.config = _StubConfig()
        self.deleted_ids = set()
        self.index_to_docstore_id = {0: "doc-1"}
        self.docstore = {
            "doc-1": Chunk(id="doc-1", content="hello", owner_id="owner-1", metadata={"source": "docstore"})
        }


def test_vector_search_with_faiss_returns_chunk_copies():
    index = _StubFaissVectorDB()
    embedding = [0.0, 0.0, 0.0]

    results = RetrievalHelper.vector_search_with_faiss(index, embedding, {"k": 1, "metric": "cosine"})
    assert len(results) == 1

    returned_chunk, _score = results[0]
    returned_chunk.metadata["mutated"] = True

    assert "mutated" not in index.docstore["doc-1"].metadata
    assert index.docstore["doc-1"].metadata["source"] == "docstore"

