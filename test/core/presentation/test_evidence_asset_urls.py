from encapsulation.data_model.schema import Chunk

from core.presentation.evidence import build_chat_evidence, build_deepsearch_evidence


def test_chat_evidence_rewrites_mineru_image_urls_and_adds_document_url():
    file_id = "file-123"
    chunk = Chunk(
        id="c1",
        owner_id="owner-1",
        content="![cap](images/demo.jpg)",
        metadata={"source_file_id": file_id, "semantic_unit_type": "image", "image_urls": ["images/demo.jpg"]},
    )
    payload = build_chat_evidence([chunk], max_chunks=1)
    entry = payload["chunks"][0]
    assert entry["file_id"] == file_id
    assert entry["document_url"] == f"/knowledge/{file_id}/download"
    assert "mineru-assets" in entry["content"]
    assert entry["metadata"]["image_urls_rel"] == ["images/demo.jpg"]
    assert entry["metadata"]["image_urls"][0] == f"/knowledge/{file_id}/mineru-assets/images/demo.jpg"


def test_deepsearch_evidence_rewrites_mineru_image_urls_and_adds_document_url():
    file_id = "file-456"
    deep_payload = {
        "report": {
            "evidences": [
                {
                    "chunk_id": "c2",
                    "source": "dense",
                    "content": "![cap](images/demo.jpg)",
                    "score": 0.9,
                    "provenance": {
                        "metadata": {
                            "chunk_metadata": {"source_file_id": file_id, "image_urls": ["images/demo.jpg"]}
                        }
                    },
                }
            ]
        },
        "reasoning": {"evidences": []},
    }
    evidence = build_deepsearch_evidence(deep_payload, chunk_limit=1)
    entry = evidence["chunks"][0]
    assert entry["file_id"] == file_id
    assert entry["document_url"] == f"/knowledge/{file_id}/download"
    assert entry["content"] == f"![cap](/knowledge/{file_id}/mineru-assets/images/demo.jpg)"
