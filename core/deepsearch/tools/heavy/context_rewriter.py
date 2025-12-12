"""Tool that rewrites context windows for long-horizon reasoning."""
from typing import Any, List

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.prompts.deepsearch import CONTEXT_ROLLUP_PROMPT

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async


class ContextRewriterTool(GraphTool):
    """Produces rewritten context emphasising unresolved entities/relations."""

    descriptor = ToolDescriptor(
        name="graph.context_rewriter",
        channel="graph",
        description="Rewrites the active context window into an explicit checklist of unresolved entities, "
        "relations, and follow-up questions so later tools focus on the remaining gaps.",
        speed="slow",
        cost="high",
        strategy_tags=("rewrite", "context", "llm"),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.heavy.context_rewriter",
        mcp_callable=True,
        example_args={
            "question": "Summarize the outstanding issues",
            "plan_step": "plan_05",
            "context_evidences": [
                {"chunk_id": "c1", "source": "hipporag", "content": "Issue 1"},
                {"chunk_id": "c2", "source": "hipporag", "content": "Issue 2"},
            ],
        },
    )

    def __init__(self, llm_connector, *, window_size: int = 8, temperature: float = 0.2):
        self.llm_connector = llm_connector
        self.window_size = window_size
        self.temperature = temperature

    async def run(self, request: ToolRunRequest) -> ToolResult:
        evidences = request.context_evidences[-self.window_size :]
        if not evidences:
            return ToolResult(summary="Context rewriter skipped because no evidences are available.")
        rewritten = await self._rewrite(request, evidences)
        evidence_chunk = EvidenceChunk(
            chunk_id="context-rewriter-0",
            source="context_rewriter",
            content=rewritten,
            provenance={"window_size": len(evidences)},
        )
        diagnostics = {
            "window_size": len(evidences),
            "token_breakdown": self._token_breakdown(evidences, rewritten),
            "thought_log": self._build_thought_log(request.plan_step, len(evidences)),
        }
        return ToolResult(summary="Context rewritten to emphasise gaps.", evidences=[evidence_chunk], diagnostics=diagnostics)

    async def _rewrite(self, request: ToolRunRequest, evidences: List[EvidenceChunk]) -> str:
        snippets = "\n\n".join(ev.content[:400] for ev in evidences)
        messages = [
            {"role": "system", "content": CONTEXT_ROLLUP_PROMPT},
            {
                "role": "user",
                "content": f"Question: {request.question}\n\nContext to rewrite:\n{snippets}\n"
                "Highlight unresolved entities and relations explicitly.",
            },
        ]
        try:
            response = await call_llm_async(self.llm_connector, messages, temperature=self.temperature)
            return response.strip()
        except Exception:
            return snippets[:800]

    @staticmethod
    def _token_breakdown(evidences: List[EvidenceChunk], rewritten_text: str) -> dict[str, int]:
        deterministic_tokens = sum(len(ev.content.split()) for ev in evidences)
        llm_tokens = len(rewritten_text.split()) if rewritten_text else 0
        return {
            "deterministic_tokens": deterministic_tokens,
            "llm_tokens": llm_tokens,
        }

    @staticmethod
    def _build_thought_log(plan_step: str | None, window_size: int) -> List[dict[str, Any]]:
        if not plan_step:
            plan_step_ref = None
        else:
            plan_step_ref = plan_step
        entry = {
            "plan_step": plan_step_ref,
            "reasoning": "Context rewritten to highlight unresolved entities and relations.",
            "reasoning_tags": ["context_rewriter"],
            "window_size": window_size,
        }
        return [entry]
