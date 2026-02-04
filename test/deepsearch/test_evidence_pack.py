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

    ids = ["chunk_001", "chunk_002"]
    source_key_map = bank.source_key_map_for_prompt(ids)
    pack = bank.evidence_pack_for_prompt(ids, source_key_map=source_key_map)
    assert "Evidence Pack" in pack
    assert "Citable Source key allowlist:" in pack
    assert "1, 2" in pack
    assert "Source key=1 id=chunk_001" in pack
    assert "Source key=2 id=chunk_002" in pack
    assert "filename=policy.pdf" in pack
    assert "page=3" in pack
    assert "filename=brochure.pdf" in pack
    # External-memory policy: evidence packs include full evidence text (no truncation).
    assert "deductible is 5000 HKD" in pack
    assert "premium refund rate" in pack

    # Bounded by item count: pass fewer ids.
    pack_one = bank.evidence_pack_for_prompt(["chunk_001"], source_key_map={"1": "chunk_001"})
    assert "chunk_002" not in pack_one
