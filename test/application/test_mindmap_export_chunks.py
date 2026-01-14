"""Test that chunks field is correctly added to nodes in mindmap export."""
import pytest

from application.knowledge.mindmap_export import export_file_mindmap_payload


class _StubKnowledge:
    async def get_file_chunk_mindmaps(self, file_id, user_id):  # noqa: ANN001, ARG002
        return {
            "file_id": file_id,
            "filename": "test_document.pdf",
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "chunk_index": 0,
                    "content": "This is the first chunk content. It contains important information about the topic.",
                    "mindmap": {},
                },
                {
                    "chunk_id": "chunk-2",
                    "chunk_index": 1,
                    "content": "This is the second chunk content. It provides additional details.",
                    "mindmap": {},
                },
                {
                    "chunk_id": "chunk-3",
                    "chunk_index": 2,
                    "content": "This is the third chunk content. It concludes the discussion.",
                    "mindmap": {},
                },
            ],
        }


class _StubLLM:
    def chat(self, _messages):  # noqa: ANN001
        return "1\tTest Document\n1.1\tSection A\n1.1.1\tDetail A1\n1.1.2\tDetail A2\n1.2\tSection B\n1.2.1\tDetail B1\n"


class _StubRagInference:
    llm = _StubLLM()


@pytest.mark.asyncio
async def test_mindmap_export_nodes_have_chunks_field() -> None:
    """Test that all nodes in the exported mindmap have chunks field as source evidence."""
    payload = await export_file_mindmap_payload(
        knowledge=_StubKnowledge(),
        rag_inference=_StubRagInference(),
        file_id="test-file-1",
        owner_id="test-owner-1",
    )

    assert isinstance(payload, dict)
    assert payload.get("tsv")
    nodes = payload.get("nodes")
    assert nodes is not None
    assert isinstance(nodes, list)
    assert len(nodes) > 0

    # Verify all nodes have chunks field
    for node in nodes:
        assert "chunks" in node, f"Node {node.get('id')} should have chunks field"
        chunks = node["chunks"]
        assert isinstance(chunks, list), "chunks should be a list"
        assert len(chunks) > 0, "chunks list should not be empty"

        # Verify chunks format
        for chunk in chunks:
            assert "id" in chunk, "Each chunk should have id field"
            assert "title" in chunk, "Each chunk should have title field"
            assert "content" in chunk, "Each chunk should have content field"
            assert "documentName" in chunk, "Each chunk should have documentName field"

            # Verify chunk id format (c1, c2, c3, ...)
            assert chunk["id"].startswith("c"), f"Chunk id should start with 'c', got: {chunk['id']}"
            
            # Verify title format (片段1, 片段2, ...)
            assert chunk["title"].startswith("片段"), f"Chunk title should start with '片段', got: {chunk['title']}"
            
            # Verify documentName matches filename
            assert chunk["documentName"] == "test_document.pdf", \
                f"documentName should be 'test_document.pdf', got: {chunk['documentName']}"


@pytest.mark.asyncio
async def test_mindmap_export_chunks_content_matches_input() -> None:
    """Test that chunks content in nodes matches the input chunks."""
    knowledge_stub = _StubKnowledge()
    payload = await export_file_mindmap_payload(
        knowledge=knowledge_stub,
        rag_inference=_StubRagInference(),
        file_id="test-file-1",
        owner_id="test-owner-1",
    )

    nodes = payload.get("nodes")
    assert nodes is not None and len(nodes) > 0

    # Get input chunks
    file_mindmaps = await knowledge_stub.get_file_chunk_mindmaps("test-file-1", "test-owner-1")
    input_chunks = file_mindmaps.get("chunks", [])

    # Verify chunks content matches for the first node
    first_node = nodes[0]
    assert "chunks" in first_node
    node_chunks = first_node["chunks"]

    assert len(node_chunks) == len(input_chunks), \
        f"Number of chunks should match. Expected {len(input_chunks)}, got {len(node_chunks)}"

    # Verify each chunk's content matches
    for idx, (node_chunk, input_chunk) in enumerate(zip(node_chunks, input_chunks), start=1):
        assert node_chunk["id"] == f"c{idx}", \
            f"Chunk id should be 'c{idx}', got: {node_chunk['id']}"
        assert node_chunk["title"] == f"片段{idx}", \
            f"Chunk title should be '片段{idx}', got: {node_chunk['title']}"
        assert node_chunk["content"] == input_chunk["content"], \
            f"Chunk content should match input chunk content"
        assert node_chunk["documentName"] == "test_document.pdf", \
            f"documentName should be 'test_document.pdf', got: {node_chunk['documentName']}"


@pytest.mark.asyncio
async def test_mindmap_export_all_nodes_have_same_chunks() -> None:
    """Test that all nodes have the same chunks (since they all come from the same source)."""
    payload = await export_file_mindmap_payload(
        knowledge=_StubKnowledge(),
        rag_inference=_StubRagInference(),
        file_id="test-file-1",
        owner_id="test-owner-1",
    )

    nodes = payload.get("nodes")
    assert nodes is not None and len(nodes) > 0

    # Get chunks from first node
    first_node_chunks = nodes[0].get("chunks")
    assert first_node_chunks is not None

    # Verify all nodes have the same chunks
    for node_idx, node in enumerate(nodes):
        node_chunks = node.get("chunks")
        assert node_chunks is not None, f"Node {node_idx} should have chunks field"
        assert len(node_chunks) == len(first_node_chunks), \
            f"Node {node_idx} should have same number of chunks as first node"

        # Verify chunks content is identical
        for chunk_idx, (node_chunk, first_chunk) in enumerate(zip(node_chunks, first_node_chunks)):
            assert node_chunk == first_chunk, \
                f"Node {node_idx}, chunk {chunk_idx} should match first node's chunk"
