"""LLM-first DeepSearch tool: chain-of-exploration."""
import json
from dataclasses import asdict
from typing import Any, Dict, List

from encapsulation.data_model.deepsearch import EvidenceChunk, ThinkNote
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED
from core.prompts.deepsearch import LLM_CHAIN_EXPLORER_SYSTEM_PROMPT_EN
from core.graph_adapter.concurrency import adapter_locked
from core.deepsearch.utils.file_scope import resolve_file_scope
from core.deepsearch.utils.evidence_ids import derived_chunk_id
from core.deepsearch.utils.evidence_cards import evidence_cards
from core.utils.llm_json import repair_json_from_raw_with_retry

from ..base import GraphTool, ToolDescriptor, ToolResult, ToolRunRequest, call_llm_async, safe_json_loads
from ..governance_tags import EVIDENCE_DERIVED, REQUIRES_LLM, SCOPE_FILE, SCOPE_OWNER


class LLMChainExplorerTool(GraphTool):
    """Delegates exploration decisions to an LLM, similar to GraphRAG heavy mode."""

    descriptor = ToolDescriptor(
        name="graph.llm_chain_explorer",
        channel="graph",
        description=(
            "LLM-managed chain-of-exploration that decides next graph hops. "
            "Evidence: derived hop summaries (NOT citeable; use deterministic graph tools for citations)."
        ),
        speed="slow",
        cost="high",
        strategy_tags=("llm", "chain_of_exploration", "hetero_ready", EVIDENCE_DERIVED, SCOPE_OWNER, SCOPE_FILE, REQUIRES_LLM),
        profile="H",
        determinism="llm_heavy",
        namespace="rag-arc.deepsearch.tools.explore.llm_chain",
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
        system_prompt: str = LLM_CHAIN_EXPLORER_SYSTEM_PROMPT_EN,
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
            raw_query = spec.get("query") or spec.get("description") or request.question
            if isinstance(raw_query, list):
                raw_query = " ".join(str(item) for item in raw_query if str(item).strip())
            sub_query = str(raw_query or request.question).strip() or str(request.question)
            channel = str(spec.get("channel") or "graph").strip() or "graph"
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
                kind=EVIDENCE_KIND_DERIVED,
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
        raw = await call_llm_async(
            self.llm_connector,
            messages,
            temperature=self.temperature,
        )
        plan = self._parse_response(raw)
        if not plan:
            repaired = await repair_json_from_raw_with_retry(
                llm_connector=self.llm_connector,
                messages=messages,
                raw=str(raw or ""),
                expected=None,
                temperature=self.temperature,
                max_tokens=None,
                include_today_line=True,
            )
            plan = self._parse_response(repaired)
        if not plan:
            fallback_query = (request.extra.get("focus_query") or request.question or "").strip()
            if not fallback_query:
                fallback_query = "graph_followup"
            plan = [
                {
                    "query": fallback_query,
                    "rationale": "fallback: model output was not a usable JSON plan; using a single hop",
                    "channel": "graph",
                    "priority": 1.0,
                    "fallback": True,
                }
            ]
        return plan

    def _build_prompt(self, request: ToolRunRequest) -> str:
        adapter_meta = asdict(request.adapter.metadata()) if request.adapter else {}
        # External-memory contract: do not inline full evidence text in tool prompts.
        # Provide a small set of metadata-only cards as context.
        cards = evidence_cards(request.context_evidences or [])
        cards = cards[-3:] if len(cards) > 3 else cards
        return json.dumps(
            {
                "question": request.question,
                "adapter_metadata": adapter_meta,
                "context_evidences": cards,
                "instructions": {
                    "max_queries": self.max_queries,
                    "allowed_channels": ["graph", "text"],
                },
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _parse_response(response: Any) -> List[Dict[str, Any]]:
        data = response
        if isinstance(response, str):
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
