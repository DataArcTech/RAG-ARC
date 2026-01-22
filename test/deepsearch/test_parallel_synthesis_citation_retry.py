import asyncio
import json


class _SynthesisLLM:
    def __init__(self) -> None:
        self.calls: int = 0

    async def achat(self, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        # First attempt: valid JSON but missing citations.
        if self.calls == 1:
            return json.dumps(
                {
                    "title": "T",
                    "short_answer": "The answer is supported by evidence.",
                    "limitations": [],
                    "next_steps": [],
                },
                ensure_ascii=False,
            )
        # Second attempt: add supported inline citation.
        return json.dumps(
            {
                "title": "T",
                "short_answer": "The answer is supported by evidence. <sup>1</sup>",
                "limitations": [],
                "next_steps": [],
            },
            ensure_ascii=False,
        )


def test_parallel_synthesis_retries_when_short_answer_missing_citations():
    from core.deepsearch.report.llm_writer import DeepSearchLLMReportWriter

    llm = _SynthesisLLM()
    writer = DeepSearchLLMReportWriter(llm_connector=llm, max_retries=2)

    async def _run():
        result = await writer._synthesize_parallel_fields(
            question="Q",
            outline=[{"title": "S", "purpose": "p", "evidence_ids": ["ev1"]}],
            sections=[{"title": "S", "section_type": "narrative", "body_markdown": "Body. <sup>1</sup>"}],
            evidences=[{"chunk_id": "ev1", "source": "local", "content": "Evidence", "score": 1.0}],
            source_key_map={"1": "ev1"},
            coverage={},
            context={},
        )
        assert llm.calls == 2
        assert "<sup>1</sup>" in str(result.get("short_answer") or "")

    asyncio.run(_run())
