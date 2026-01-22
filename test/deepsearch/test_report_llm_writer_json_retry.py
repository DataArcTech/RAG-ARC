import pytest

from core.deepsearch.memory import EvidenceBank
from core.deepsearch.report.llm_writer import DeepSearchLLMReportWriter


@pytest.mark.asyncio
async def test_section_writer_retries_on_invalid_json_then_succeeds():
    writer = DeepSearchLLMReportWriter(
        llm_connector=object(),
        temperature=0.0,
        max_retries=2,
        max_evidence_items=5,
        max_evidence_chars=800,
        max_graph_chain_items=10,
        parallel_sections=False,
        max_parallel_sections=1,
        max_section_evidence_items=3,
        synthesis_section_max_chars=800,
    )

    # First response: looks like JSON but is invalid because body_markdown contains a raw newline.
    # Second response: valid JSON and includes required inline citation.
    responses = [
        '```json\n{"title":"Direct Answer","section_type":"narrative","body_markdown":"Line1\nLine2. <sup>1</sup>","citations":[{"evidence_id":"chunk_001","used_for":"support"}]}\n```',
        '{"title":"Direct Answer","section_type":"narrative","body_markdown":"Line1\\\\nLine2. <sup>1</sup>","citations":[{"evidence_id":"chunk_001","used_for":"support"}]}',
    ]

    async def fake_call(self, messages, *, phase: str):  # noqa: ARG001
        return responses.pop(0)

    writer._call = fake_call.__get__(writer, DeepSearchLLMReportWriter)

    bank = EvidenceBank()
    bank.add_many([{"chunk_id": "chunk_001", "content": "Guaranteed Preferential Interest Rate: 3%"}])
    source_key_map = bank.source_key_map_for_prompt(bank.ids())
    evidence_pack = bank.evidence_pack_for_prompt(bank.ids(), source_key_map=source_key_map)

    result = await writer._write_single_section(
        question="What is the guaranteed preferential interest rate?",
        section={"title": "Direct Answer", "section_type": "narrative", "purpose": "Answer directly."},
        evidence_pack=evidence_pack,
        allowed_keys={1},
        graph_chain=[],
        context={},
    )
    assert result["title"] == "Direct Answer"
    assert "<sup>1</sup>" in result["body_markdown"]
