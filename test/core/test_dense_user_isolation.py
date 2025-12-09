from types import SimpleNamespace
from unittest.mock import patch
import uuid

from encapsulation.data_model.schema import Chunk
from core.retrieval.dense import DenseRetriever


def _make_dense_retriever():
    """Create a lightweight DenseRetriever instance for unit testing."""
    retriever = DenseRetriever.__new__(DenseRetriever)
    retriever.config = SimpleNamespace(search_kwargs={"k": 2}, metric="cosine")
    retriever._index = SimpleNamespace(
        index=SimpleNamespace(ntotal=50),
        config=SimpleNamespace(normalize_L2=False, metric="cosine"),
    )
    retriever.embedding = SimpleNamespace(embed=lambda _: [0.1, 0.2])
    return retriever


def test_similarity_search_filters_uuid_and_string_owners():
    retriever = _make_dense_retriever()
    owner_uuid = uuid.uuid4()
    other_uuid = uuid.uuid4()

    chunk_for_owner = Chunk(id="chunk-a", content="alpha", owner_id=owner_uuid)
    chunk_for_other = Chunk(id="chunk-b", content="beta", owner_id=str(other_uuid))

    mock_results = [(chunk_for_owner, 0.95), (chunk_for_other, 0.9)]
    with patch("core.retrieval.dense.RetrievalHelper.vector_search_with_faiss", return_value=mock_results):
        chunks = retriever.similarity_search_by_vector(
            embedding=[0.0, 0.0],
            owner_id=str(owner_uuid),
            k=2,
        )

    assert chunks == [chunk_for_owner]


def test_similarity_search_falls_back_to_metadata_owner():
    retriever = _make_dense_retriever()
    owner_uuid = uuid.uuid4()

    chunk_with_metadata_owner = Chunk(
        id="chunk-meta",
        content="via metadata",
        owner_id=None,
        metadata={"owner_id": str(owner_uuid)},
    )
    chunk_without_owner = Chunk(id="chunk-none", content="no owner", owner_id=None)

    mock_results = [(chunk_with_metadata_owner, 0.9), (chunk_without_owner, 0.8)]
    with patch("core.retrieval.dense.RetrievalHelper.vector_search_with_faiss", return_value=mock_results):
        chunks = retriever.similarity_search_by_vector(
            embedding=[0.0, 0.0],
            owner_id=owner_uuid,
            k=2,
        )

    assert chunks == [chunk_with_metadata_owner]


def test_mmr_search_filters_by_owner_before_fusion():
    retriever = _make_dense_retriever()
    owner_uuid = uuid.uuid4()
    chunk_for_owner = Chunk(id="mmr-chunk", content="owner", owner_id=str(owner_uuid))
    chunk_for_other = Chunk(id="mmr-other", content="other", owner_id=str(uuid.uuid4()))

    mock_results = [(chunk_for_owner, 0.9), (chunk_for_other, 0.8)]

    captured_inputs = []

    def fake_mmr_search(query_embedding, chunks_and_scores, embedding_model, search_kwargs):
        captured_inputs.append(chunks_and_scores)
        return [chunk for chunk, _ in chunks_and_scores]

    with patch("core.retrieval.dense.RetrievalHelper.vector_search_with_faiss", return_value=mock_results), patch(
        "core.retrieval.dense.RetrievalHelper.mmr_search",
        side_effect=fake_mmr_search,
    ):
        chunks = retriever.max_marginal_relevance_search(
            query="test",
            owner_id=owner_uuid,
            k=1,
        )

    assert chunks == [chunk_for_owner]
    assert captured_inputs == [[(chunk_for_owner, 0.9)]]


def test_dense_retriever_rejects_missing_owner():
    retriever = _make_dense_retriever()
    chunks = retriever.similarity_search_by_vector(embedding=[0.0, 0.0], owner_id=None)
    assert chunks == []
