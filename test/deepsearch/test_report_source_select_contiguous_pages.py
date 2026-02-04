from core.deepsearch.report.llm_writer import _expand_contiguous_pages  # noqa: SLF001 (unit-level helper)
from core.deepsearch.memory import EvidenceBank


def _ev(*, page: int, file_id: str = "f1", chunk_id: str | None = None) -> dict:
    return {
        "chunk_id": chunk_id or f"e{page}",
        "source": "read.pages",
        "content": f"page={page} table row 1.23% IRR 4.20",  # include numeric signal
        "provenance": {"source_file_id": file_id, "page_start": page, "page_end": page, "metadata": {"filename": "doc.pdf"}},
    }


def test_expand_contiguous_pages_prefers_neighbors() -> None:
    bank = EvidenceBank()
    evidences = [_ev(page=p) for p in range(10, 15)]  # 10..14
    bank.add_many(evidences)
    all_ids = [e["chunk_id"] for e in evidences]

    # Model selects the middle page only; we should expand to neighbors first.
    selected = _expand_contiguous_pages(bank=bank, candidate_ids=all_ids, selected_ids=["e12"], max_sources=4)
    assert selected[0] == "e12"
    assert set(selected) == {"e12", "e11", "e13", "e10"} or set(selected) == {"e12", "e11", "e13", "e14"}


def test_expand_contiguous_pages_does_not_exceed_budget() -> None:
    bank = EvidenceBank()
    evidences = [_ev(page=p) for p in range(1, 20)]
    bank.add_many(evidences)
    all_ids = [e["chunk_id"] for e in evidences]

    selected = _expand_contiguous_pages(bank=bank, candidate_ids=all_ids, selected_ids=["e10"], max_sources=3)
    assert len(selected) == 3

