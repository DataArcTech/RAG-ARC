"""Prompt templates for DeepSearch tool executors."""

EVIDENCE_CROSSCHECK_PROMPT = (
    "You validate chunk summaries against candidate knowledge graph triples before the report stage.\n"
    "\n"
    "Given the question, chunk snippets, and triples:\n"
    "- Mark each triple as SUPPORTED when at least one chunk explicitly backs it.\n"
    "- Mark as UNSUPPORTED when no retrieved chunk justifies the relation.\n"
    "- Use grounded reasoning only; do not invent new nodes or predicates.\n"
    "- Respond strictly with JSON in the form:\n"
    "  {\n"
    '    "supported": [\n'
    '      {"triple": "entity1 -[relation]-> entity2", "chunks": ["chunk_id"], "reason": "..."}\n'
    "    ],\n"
    '    "unsupported": [\n'
    '      {"triple": "...", "reason": "..."}\n'
    "    ],\n"
    '    "summary": "textual overview"\n'
    "  }\n"
    "Keep the summary concise and focus on actionable guidance for the reasoning loop."
)

LLM_CHAIN_EXPLORER_SYSTEM_PROMPT = (
    "You are a graph exploration planner that emits JSON instructions for DeepSearch traversal. "
    "Reason step by step, cite why each hop is useful, and stay within allowed channels."
)

CONTEXT_ROLLUP_PROMPT = (
    "You compress graph evidences into structured bullets for DeepSearch planning. "
    "Summaries must stay faithful to the snippets and highlight follow-up leads when relevant."
)

CONTEXT_REWRITER_PROMPT = (
    "You rewrite the provided evidence window into a concise checklist:\n"
    "- unresolved entities\n"
    "- unresolved relations\n"
    "- concrete follow-up questions\n"
    "Do not invent facts that are not supported by the snippets."
)

THINK_TOOL_SYSTEM_PROMPT = (
    "You are a reflection module for a graph-first research agent.\n"
    "\n"
    "Task:\n"
    "- Read the provided question, evidence chunks, and coverage signals.\n"
    "- Produce a concise reasoning summary and propose concrete next actions.\n"
    "\n"
    "Critical constraints:\n"
    "- Do NOT reveal step-by-step hidden chain-of-thought. Provide a short, user-facing rationale only.\n"
    "- Do NOT invent facts not supported by evidence.\n"
    "- Keep next actions executable (graph-first; use web only when evidence is insufficient).\n"
    "\n"
    "Return ONLY valid JSON with keys:\n"
    "- reasoning: string\n"
    "- confidence_delta: number | null\n"
    "- coverage_delta: number | null\n"
    "- next_actions: array of strings\n"
    "- gap_trigger: boolean (true if external search should be considered)\n"
    "- missing_topics: array of strings\n"
)

PARALLEL_THINK_SYSTEM_PROMPT = (
    "Generate multiple reasoning branches as JSON. Each item must include:\n"
    '- "thought": a short hypothesis or angle to explore\n'
    '- "action": a concrete next tool/step name or action label\n'
)

HYBRID_NEIGHBORHOOD_SUMMARY_PROMPT = (
    "You condense chunk evidence into concise reasoning bullets.\n"
    "Only use the provided snippets; do not fabricate citations or entities."
)
