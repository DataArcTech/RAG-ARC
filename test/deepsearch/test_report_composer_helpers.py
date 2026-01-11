from core.deepsearch.report.composer_helpers import _split_authoritative_evidences


def test_split_authoritative_evidences_does_not_treat_colon_chunk_ids_as_tool_generated():
    authoritative, generated = _split_authoritative_evidences(
        [
            {"chunk_id": "file:12:3", "source": "hipporag", "content": "primary"},
            {"chunk_id": "graph.think:1", "source": "think", "content": "toolish"},
        ]
    )

    assert len(authoritative) == 1
    assert authoritative[0]["chunk_id"] == "file:12:3"
    assert any(ev.get("chunk_id") == "graph.think:1" for ev in generated)

