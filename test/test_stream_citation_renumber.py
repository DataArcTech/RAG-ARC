from api.routers.chatbot import ChatbotSourceItem


def _make_source(key: int) -> ChatbotSourceItem:
    return ChatbotSourceItem(
        key=key,
        chunk_id=f"chunk-{key}",
        title=f"title-{key}",
        description=f"desc-{key}",
    )


def test_streaming_sup_renumber_first_appearance_order():
    from api.routers.rag_inference_modules.stream_chat.utils.citation_stream import (
        CitationStreamRenumberer,
    )

    renumberer = CitationStreamRenumberer()
    pieces = [
        "A<sup>6",
        "</sup> B<sup>2</sup>",
        " C<sup>6</sup>",
    ]
    output = "".join(renumberer.feed(piece) for piece in pieces) + renumberer.flush()

    assert output == "A<sup>1</sup> B<sup>2</sup> C<sup>1</sup>"
    assert renumberer.key_map == {6: 1, 2: 2}


def test_sources_renumber_by_first_appearance_order():
    from api.routers.chatbot import _filter_and_renumber_sources_by_sup_keys_appearance

    sources = [_make_source(1), _make_source(2), _make_source(6), _make_source(20)]
    answer = "X<sup>6</sup> Y<sup>2</sup> Z<sup>20</sup>"

    new_answer, new_sources, key_map = _filter_and_renumber_sources_by_sup_keys_appearance(
        sources,
        answer,
    )

    assert new_answer == "X<sup>1</sup> Y<sup>2</sup> Z<sup>3</sup>"
    assert [s.chunk_id for s in new_sources] == ["chunk-6", "chunk-2", "chunk-20"]
    assert [s.key for s in new_sources] == [1, 2, 3]
    assert key_map == {6: 1, 2: 2, 20: 3}


def test_streaming_sup_renumber_ignores_out_of_range_keys():
    from api.routers.rag_inference_modules.stream_chat.utils.citation_stream import (
        CitationStreamRenumberer,
    )

    renumberer = CitationStreamRenumberer(max_key=5)
    out = renumberer.feed("A<sup>8</sup>B<sup>2</sup>C") + renumberer.flush()

    assert out == "AB<sup>1</sup>C"
    assert renumberer.key_map == {2: 1}


def test_sources_include_page_range_when_available():
    from api.routers.chatbot import _build_sources_for_frontend_with_llm_keys

    entries = [
        {
            "id": "chunk-1",
            "content": "hello",
            "metadata": {
                "source_file_id": "file-1",
                "filename": "a.pdf",
                "page_start": 3,
                "page_end": 4,
            },
        }
    ]
    sources = _build_sources_for_frontend_with_llm_keys(entries, 5, {"chunk-1": 1})
    assert len(sources) == 1
    assert sources[0].page_start == 3
    assert sources[0].page_end == 4
