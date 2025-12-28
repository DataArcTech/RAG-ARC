"""LLM-first DeepSearch tool: chain-of-exploration."""
import json
from dataclasses import asdict
from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk, ThinkNote
from core.prompts.deepsearch import LLM_CHAIN_EXPLORER_SYSTEM_PROMPT
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.file_scope import resolve_file_scope
from core.deepsearch.utils.evidence_ids import derived_chunk_id

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, safe_json_loads


class LLMChainExplorerTool(GraphTool):
    """Delegates exploration decisions to an LLM, similar to GraphRAG heavy mode."""

    descriptor = ToolDescriptor(
        name="graph.llm_chain_explorer",
        channel="graph",
        description="LLM-managed chain-of-exploration that decides next graph hops.",
        speed="slow",
        cost="high",
        strategy_tags=("llm", "chain_of_exploration", "hetero_ready"),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.heavy.llm_chain",
        mcp_callable=True,
        example_args={
            "question": "Explain OpenAI's origin",
            "plan_step": "plan_02",
            "context_evidences": [
                {"chunk_id": "c1", "source": "hipporag", "content": "OpenAI founded in 2015."}
            ],
        },
    )

    def __init__(
        self,
        llm_connector,
        *,
        max_queries: int = 4,
        temperature: float = 0.2,
        system_prompt: str = LLM_CHAIN_EXPLORER_SYSTEM_PROMPT,
    ):
        self.llm_connector = llm_connector
        self.max_queries = max_queries
        self.temperature = temperature
        self.system_prompt = system_prompt

    async def run(self, request: ToolRunRequest) -> ToolResult:
        plan = await self._generate_plan(request)
        adapter = request.adapter
        if adapter is None:
            raise RuntimeError("LLMChainExplorerTool requires a GraphDeepSearchAdapter")

        file_scope = resolve_file_scope(
            extra=request.extra,
            graph_context_metadata=(request.graph_context.metadata if request.graph_context else {}),
            question=request.question,
        )
        evidences: List[EvidenceChunk] = []
        diagnostics = {
            "plan": plan,
            "thought_log": self._build_thought_log(plan, request.plan_step),
        }
        if file_scope.enabled:
            diagnostics["file_scope"] = file_scope.as_dict()
        for idx, spec in enumerate(plan[: self.max_queries]):
            sub_query = spec.get("query") or spec.get("description") or request.question
            channel = spec.get("channel") or "graph"
            async with adapter_locked(adapter):
                payload = await adapter.aquery_subgraph(
                    sub_query,
                    channel=channel,
                    access_scope=request.access_scope,
                    query_options={"file_scope": file_scope.as_dict()} if file_scope.enabled else None,
                )
                summary = await adapter.summarize(channel, payload, access_scope=request.access_scope)
            evidence = EvidenceChunk(
                chunk_id=derived_chunk_id(
                    tool_name=self.descriptor.name,
                    plan_step=request.plan_step,
                    label=f"llm_chain_{idx}",
                    content=str(summary),
                ),
                source=self.descriptor.name,
                content=summary,
                score=spec.get("priority", 1.0),
                provenance={
                    "llm_plan_step": spec,
                    "raw_payload": payload,
                },
            )
            evidences.append(evidence)

        summary_lines = [
            f"{idx+1}. {spec.get('query')} ({spec.get('rationale')})" for idx, spec in enumerate(plan[: self.max_queries])
        ]
        summary = "LLM chain explorer executed the following hops:\n" + "\n".join(summary_lines)
        think_note = self._build_think_note(request, plan, evidences)
        return ToolResult(summary=summary, evidences=evidences, diagnostics=diagnostics, think_notes=think_note)

    async def _generate_plan(self, request: ToolRunRequest) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(request)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        response = await call_llm_async(
            self.llm_connector,
            messages,
            temperature=self.temperature,
        )
        plan = self._parse_response(response)
        if not plan:
            raise ValueError("LLM chain explorer returned an empty plan")
        return plan

    def _build_prompt(self, request: ToolRunRequest) -> str:
        adapter_meta = asdict(request.adapter.metadata()) if request.adapter else {}
        context_snippets = "\n".join(e.content[:200] for e in request.context_evidences[-3:])
        return json.dumps(
            {
                "question": request.question,
                "adapter_metadata": adapter_meta,
                "context": context_snippets,
                "instructions": {
                    "max_queries": self.max_queries,
                    "allowed_channels": ["graph", "text"],
                },
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _parse_response(response: str) -> List[Dict[str, Any]]:
        data = safe_json_loads(response)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and "steps" in data:
            steps = data["steps"]
            if isinstance(steps, list):
                return [item for item in steps if isinstance(item, dict)]
        return []

    def _build_think_note(
        self,
        request: ToolRunRequest,
        plan: List[Dict[str, Any]],
        evidences: List[EvidenceChunk],
    ) -> List[ThinkNote]:
        coverage = min(1.0, len(evidences) / max(1, self.max_queries))
        actions = [step.get("query", "graph_followup") for step in plan[:2] if step.get("query")]
        note = ThinkNote(
            plan_step_id=request.plan_step,
            reasoning="LLM chain explorer reviewed the plan and suggested follow-up hops.",
            confidence_delta=coverage - 0.5,
            coverage_delta=coverage,
            next_actions=actions,
            metadata={"plan_steps": len(plan)},
        )
        return [note]

    @staticmethod
    def _build_thought_log(plan: List[Dict[str, Any]], plan_step: str | None) -> List[Dict[str, Any]]:
        log: List[Dict[str, Any]] = []
        for idx, spec in enumerate(plan):
            log.append(
                {
                    "plan_step": plan_step,
                    "hop": idx,
                    "reasoning": spec.get("rationale") or spec.get("description") or "Executed graph hop",
                    "next_action": spec.get("query"),
                    "reasoning_tags": ["llm_chain_explorer"],
                    "channel": spec.get("channel", "graph"),
                }
            )
        return log
