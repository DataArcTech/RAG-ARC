from core.deepsearch.memory import EvidenceBank
from core.deepsearch.utils.compression import focused_truncate_text


def test_focused_truncate_preserves_query_match_near_tail():
    content = ("A" * 500) + " KEY_PHRASE " + ("B" * 200)
    out = focused_truncate_text(
        content,
        max_chars=80,
        question="Where is KEY_PHRASE mentioned?",
        extra=None,
    )
    assert "KEY_PHRASE" in out
    assert len(out) <= 80


def test_focused_truncate_falls_back_when_no_match():
    content = "prefix-" + ("x" * 500) + "-suffix"
    out = focused_truncate_text(
        content,
        max_chars=40,
        question="totally absent token",
        extra=None,
    )
    assert out.startswith("prefix-")
    assert len(out) <= 40


def test_evidence_bank_select_evidences_uses_focused_truncate():
    bank = EvidenceBank()
    bank.add_many(
        [
            {
                "chunk_id": "c1",
                "source": "local",
                "content": ("x" * 400) + " IMPORTANT_FACT " + ("y" * 400),
                "score": 0.9,
            }
        ]
    )
    selected = bank.select_evidences(["c1"], max_chars=90, question="Please cite IMPORTANT_FACT.")
    assert selected and selected[0]["chunk_id"] == "c1"
    assert "IMPORTANT_FACT" in (selected[0].get("content") or "")
    assert len(str(selected[0].get("content") or "")) <= 90

