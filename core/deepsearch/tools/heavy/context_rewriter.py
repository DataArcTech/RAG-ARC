"""Tool that rewrites context windows for long-horizon reasoning."""
from typing import Any, List

from encapsulation.data_model.deepsearch import EvidenceChunk
from core.prompts.deepsearch import CONTEXT_REWRITER_PROMPT
from core.deepsearch.utils.evidence_ids import derived_chunk_id
from core.deepsearch.utils.compression import compact_context_snippet, resolve_compaction_config, truncate_text

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_OWNER


class ContextRewriterTool(GraphTool):
    """Produces rewritten context emphasising unresolved entities/relations."""

    descriptor = ToolDescriptor(
        name="graph.context_rewriter",
        channel="graph",
        description="Rewrites the active context window into an explicit checklist of unresolved entities, "
        "relations, and follow-up questions so later tools focus on the remaining gaps.",
        speed="slow",
        cost="high",
        strategy_tags=("rewrite", "context", "llm", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_LLM),
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
            chunk_id=derived_chunk_id(
                tool_name=self.descriptor.name,
                plan_step=request.plan_step,
                label="rewrite",
                content=rewritten,
            ),
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
        cfg = resolve_compaction_config(
            branch="tool_context",
            graph_context=request.graph_context,
            extra=(request.extra or {}),
            default_max_items=max(1, int(self.window_size)),
            default_max_chars=400,
            default_mode="truncate",
            default_retention="tail",
            env_max_items="DEEPSEARCH_TOOL_CONTEXT_MAX_EVIDENCES",
            env_max_chars="DEEPSEARCH_TOOL_CONTEXT_MAX_CHARS",
        )
        snippets, _meta = compact_context_snippet(
            evidences,
            cfg=cfg,
            question=request.question,
            extra=(request.extra or {}),
            joiner="\n\n",
        )
        messages = [
            {"role": "system", "content": CONTEXT_REWRITER_PROMPT},
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
            return truncate_text(snippets, max_chars=800)

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
