import pytest

from application.knowledge.mindmap_export import export_file_mindmap_payload


class _StubKnowledge:
    async def get_file_chunk_mindmaps(self, file_id, user_id):  # noqa: ANN001, ARG002
        return {
            "file_id": file_id,
            "filename": "demo.pdf",
            "chunks": [
                {"chunk_id": "c1", "chunk_index": 0, "content": "Section A: policy currency and premium payment term.", "mindmap": {}},
                {"chunk_id": "c2", "chunk_index": 1, "content": "Section B: cash value, surrender value, death benefit.", "mindmap": {}},
            ],
        }


class _StubLLM:
    def chat(self, _messages):  # noqa: ANN001
        return "1\tDemo Document\n1.1\tKey Fields\n1.1.1\tPolicy currency\n1.1.2\tPremium payment term\n"


class _StubRagInference:
    llm = _StubLLM()


@pytest.mark.asyncio
async def test_mindmap_export_does_not_require_ingest_time_chunk_mindmaps() -> None:
    payload = await export_file_mindmap_payload(
        knowledge=_StubKnowledge(),
        rag_inference=_StubRagInference(),
        file_id="file-1",
        owner_id="owner-1",
    )
    assert isinstance(payload, dict)
    assert payload.get("tsv")
    assert payload.get("nodes")
    assert payload.get("edges") is not None

