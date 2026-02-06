import uuid

from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from encapsulation.data_model.schema import Chunk
from encapsulation.database.index_scoping import owner_scoped_dir


def test_owner_scoped_bm25_retriever_loads_per_owner_index(tmp_path) -> None:
    owner_id = str(uuid.uuid4())
    base_dir = str(tmp_path / "bm25_index")

    # Build an owner-scoped index on disk (as the ingestion pipeline would).
    builder_cfg = BM25BuilderConfig(type="bm25_builder", index_path=base_dir, owner_scoped_enabled=True)
    index_dir = owner_scoped_dir(
        base_dir,
        owner_id=owner_id,
        owner_dirname=str(getattr(builder_cfg, "owner_scoped_dirname", "owners") or "owners"),
        global_owner_name=str(getattr(builder_cfg, "owner_scoped_global_owner_name", "__GLOBAL__") or "__GLOBAL__"),
    )
    builder = builder_cfg.model_copy(update={"index_path": index_dir}).build()

    ch = Chunk(
        id=str(uuid.uuid4()),
        content="Backup insured / successor insured options are described here.",
        owner_id=owner_id,
        metadata={"source_file_id": "file-1", "filename": "a.pdf"},
    )
    builder.from_chunks([ch])

    # Retriever should resolve index_dir from (base_dir, owner_id) and not touch self._index (None in owner-scoped mode).
    retr_cfg = TantivyBM25RetrieverConfig(type="tantivy_bm25", index_config=builder_cfg)
    retr = retr_cfg.build()

    out = retr.invoke("backup insured", owner_id=owner_id, k=5, with_score=True)
    assert out
    assert out[0].id == ch.id

