from core.deepsearch.report.composer_helpers import stable_sort_authoritative_evidences


def test_stable_sort_authoritative_evidences_orders_read_pages_by_file_and_page() -> None:
    items = [
        {
            "chunk_id": "rp_b_2",
            "source": "read.pages",
            "content": "p2",
            "provenance": {"source_file_id": "b", "page_start": 2, "page_end": 2, "metadata": {"filename": "b.pdf"}},
        },
        {
            "chunk_id": "other_1",
            "source": "graph.ops",
            "content": "derived",
            "provenance": {},
        },
        {
            "chunk_id": "rp_a_10",
            "source": "read.pages",
            "content": "p10",
            "provenance": {"source_file_id": "a", "page_start": 10, "page_end": 10, "metadata": {"filename": "a.pdf"}},
        },
        {
            "chunk_id": "rp_a_1",
            "source": "read.pages",
            "content": "p1",
            "provenance": {"source_file_id": "a", "page_start": 1, "page_end": 1, "metadata": {"filename": "a.pdf"}},
        },
    ]

    ordered = stable_sort_authoritative_evidences(items)
    assert [ev["chunk_id"] for ev in ordered] == ["rp_a_1", "rp_a_10", "rp_b_2", "other_1"]

