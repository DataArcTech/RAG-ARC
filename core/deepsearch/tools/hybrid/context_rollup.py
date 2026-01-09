"""LLM-backed context rollup tool for reasoning checkpoints."""
from typing import Dict, List

from config.core.deepsearch.tool_defaults import (
    CONTEXT_ROLLUP_DEFAULT_SNIPPET_CHARS,
    CONTEXT_ROLLUP_DEFAULT_TEMPERATURE,
    CONTEXT_ROLLUP_DEFAULT_WINDOW_SIZE,
)
from encapsulation.data_model.deepsearch import EvidenceChunk
from core.prompts.deepsearch import CONTEXT_ROLLUP_PROMPT
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_OWNER


class ContextRollupTool(GraphTool):
    """Compresses the active evidence window into a structured summary."""

    descriptor = ToolDescriptor(
        name="graph.context_rollup",
        channel="graph",
        description=(
            "LLM rollup of active evidence to stabilise long reasoning chains. "
            "Evidence: derived summary (NOT citeable; cite underlying chunks/tools instead)."
        ),
        speed="medium",
        cost="medium",
        strategy_tags=("summary", "context", "llm", EVIDENCE_DERIVED, SCOPE_OWNER, REQUIRES_LLM),
        profile="X",
        determinism="hybrid",
        namespace="rag-arc.deepsearch.tools.x.context_rollup",
        mcp_callable=True,
        example_args={
            "question": "Summarize the current findings",
            "plan_step": "plan_03",
            "context_evidences": [
                {"chunk_id": "c1", "source": "hipporag", "content": "Finding one"},
                {"chunk_id": "c2", "source": "hipporag", "content": "Finding two"},
            ],
        },
    )

    def __init__(
        self,
        llm_connector,
        *,
        window_size: int = CONTEXT_ROLLUP_DEFAULT_WINDOW_SIZE,
        temperature: float = CONTEXT_ROLLUP_DEFAULT_TEMPERATURE,
        system_prompt: str = CONTEXT_ROLLUP_PROMPT,
    ):
        self.llm_connector = llm_connector
        self.window_size = window_size
        self.temperature = temperature
        self.system_prompt = system_prompt

    async def run(self, request: ToolRunRequest) -> ToolResult:
        evidences = self._window(request)
        if not evidences:
            return ToolResult(
                summary="Context rollup skipped because no evidences are available.",
                diagnostics={"window_size": self.window_size},
            )
        summary_text = await self._summarize(request, evidences)
        rollup_chunk = EvidenceChunk(
            chunk_id=derived_chunk_id(
                tool_name=self.descriptor.name,
                plan_step=request.plan_step,
                label="rollup",
                content=summary_text,
            ),
            source="context_rollup",
            content=summary_text,
            score=1.0,
            provenance={"window_size": len(evidences)},
        )
        diagnostics = {
            "window_size": len(evidences),
            "temperature": self.temperature,
        }
        diagnostics["token_breakdown"] = self._token_breakdown(evidences, summary_text)
        summary = "Context rollup produced a structured digest for downstream planners."
        return ToolResult(summary=summary, evidences=[rollup_chunk], diagnostics=diagnostics)

    def _window(self, request: ToolRunRequest) -> List[EvidenceChunk]:
        if not request.context_evidences:
            return []
        return request.context_evidences[-self.window_size :]

    async def _summarize(self, request: ToolRunRequest, evidences: List[EvidenceChunk]) -> str:
        snippet_chars = int(CONTEXT_ROLLUP_DEFAULT_SNIPPET_CHARS)
        snippets = "\n\n".join(ev.content[:snippet_chars] for ev in evidences)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": f"Question: {request.question}\n\nContext:\n{snippets}",
            },
        ]
        response = await call_llm_async(
            self.llm_connector,
            messages,
            temperature=self.temperature,
        )
        rendered = (response or "").strip()
        if not rendered:
            raise RuntimeError("ContextRollupTool returned an empty response")
        return rendered

    @staticmethod
    def _token_breakdown(evidences: List[EvidenceChunk], summary_text: str) -> Dict[str, int]:
        deterministic_tokens = sum(len(ev.content.split()) for ev in evidences)
        llm_tokens = len(summary_text.split()) if summary_text else 0
        return {
            "deterministic_tokens": deterministic_tokens,
            "llm_tokens": llm_tokens,
        }
