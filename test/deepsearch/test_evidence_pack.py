from core.deepsearch.memory import EvidenceBank


def test_evidence_pack_is_deterministic_and_bounded() -> None:
    bank = EvidenceBank()
    bank.add_many(
        [
            {
                "chunk_id": "chunk_001",
                "source": "hipporag",
                "content": "A" * 200 + " deductible is 5000 HKD " + "B" * 200,
                "provenance": {"metadata": {"filename": "policy.pdf", "page": 3}},
            },
            {
                "chunk_id": "chunk_002",
                "source": "hipporag",
                "content": "C" * 80 + " premium refund rate " + "D" * 80,
                "provenance": {"metadata": {"filename": "brochure.pdf", "page_number": 1}},
            },
        ]
    )

    pack = bank.evidence_pack_for_prompt(
        ["chunk_001", "chunk_002"],
        question="What is the deductible?",
        max_chars_per_evidence=60,
        max_summary_chars=20,
    )
    assert "Evidence Pack" in pack
    assert "Citable chunk_id allowlist:" in pack
    assert "chunk_001, chunk_002" in pack
    assert "- [chunk_001]" in pack
    assert "- [chunk_002]" in pack
    assert "filename=policy.pdf" in pack
    assert "page=3" in pack
    assert "filename=brochure.pdf" in pack

    excerpt = bank.excerpt_for_prompt("chunk_001", question="What is the deductible?", max_chars=60)
    assert excerpt
    assert len(excerpt) <= 60

    pack_one = bank.evidence_pack_for_prompt(["chunk_001", "chunk_002"], max_items=1, max_chars_per_evidence=40)
    assert "chunk_002" not in pack_one

