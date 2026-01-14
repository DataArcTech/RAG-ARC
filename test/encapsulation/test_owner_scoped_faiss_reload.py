import os

import numpy as np
import pytest

import faiss

from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from encapsulation.data_model.schema import Chunk
from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin
from encapsulation.database.vector_db.faiss import FaissVectorDB


class _DummyOwnerScopedFaissStore(_PrunedHippoRAGNeo4jEmbeddingsMixin):
    def __init__(self, *, storage_path: str) -> None:
        self.storage_path = storage_path
        embedding_cfg = OpenAIEmbeddingConfig(
            loading_method="huggingface",
            model_name="dummy",
            embedding_dimensions=3,
            openai_api_key="",
            openai_base_url="",
        )
        self.fact_faiss_db = FaissVectorDBConfig(
            embedding_config=embedding_cfg,
            index_type="flat",
            metric="cosine",
            normalize_L2=True,
            index_path=os.path.join(storage_path, "template_fact"),
            index_name="index",
        )
        self.entity_faiss_db = FaissVectorDBConfig(
            embedding_config=embedding_cfg,
            index_type="flat",
            metric="cosine",
            normalize_L2=True,
            index_path=os.path.join(storage_path, "template_entity"),
            index_name="index",
        )

    def _restore_owner_id(self, owner_id: object) -> str | None:
        if owner_id is None:
            return None
        return str(owner_id)


def _write_valid_index(index_dir: str, *, owner: str, kind: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    embedding_cfg = OpenAIEmbeddingConfig(
        loading_method="huggingface",
        model_name="dummy",
        embedding_dimensions=3,
        openai_api_key="",
        openai_base_url="",
    )
    cfg = FaissVectorDBConfig(
        embedding_config=embedding_cfg,
        index_type="flat",
        metric="cosine",
        normalize_L2=True,
        index_path=index_dir,
        index_name="index",
    )
    db = FaissVectorDB(cfg)

    dim = 3
    db.index = faiss.IndexFlatIP(dim)
    vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    faiss.normalize_L2(vec)
    db.index.add(vec)

    doc_id = f"{kind}-1"
    db.docstore[doc_id] = Chunk(
        id=doc_id,
        content="a | relates_to | b",
        owner_id=owner,
        metadata={"type": kind, "owner_id": owner, "embedding": vec[0]},
    )
    db.index_to_docstore_id[0] = doc_id
    db.save_index(index_dir, "index")


@pytest.mark.parametrize("kind", ["fact", "entity"])
def test_owner_scoped_faiss_db_reloads_when_artifacts_appear(tmp_path, kind: str) -> None:
    store = _DummyOwnerScopedFaissStore(storage_path=str(tmp_path))
    owner = "00000000-0000-0000-0000-000000000001"

    index_dir = store._faiss_owner_scoped_dir(kind, owner_id=owner)
    _write_valid_index(index_dir, owner=owner, kind=kind)

    pkl_path = os.path.join(index_dir, "index.pkl")
    tmp_pkl_path = os.path.join(index_dir, "index.pkl.tmp")
    os.replace(pkl_path, tmp_pkl_path)

    # `FaissVectorDB` is a @shared_module; clear process-level cache to simulate a fresh worker
    # that hasn't loaded the index in memory yet.
    FaissVectorDB.clear_cache()

    db_1 = store.get_fact_faiss_db(owner) if kind == "fact" else store.get_entity_faiss_db(owner)
    assert db_1.index is None

    os.replace(tmp_pkl_path, pkl_path)

    db_2 = store.get_fact_faiss_db(owner) if kind == "fact" else store.get_entity_faiss_db(owner)
    assert db_2 is db_1
    assert db_2.index is not None
    assert int(db_2.index.ntotal) == 1
