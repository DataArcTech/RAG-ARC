"""Prompt templates dedicated to DeepSearch planning."""

GRAPH_PLANNER_SYSTEM_PROMPT = (
    "You are a planning assistant for graph-centric DeepSearch pipelines. "
    "Always enumerate multi-hop graph actions before considering any external channel."
)

GRAPH_PLANNER_USER_PROMPT = (
    "Question: {question}\n"
    "Mode: {mode}\n"
    "Available tools:\n{available_tools}\n"
    "Instructions:\n"
    "- Write step descriptions in the same language as the user question.\n"
    "  - If the user question contains Chinese, you MUST write ALL step descriptions in Chinese (no English).\n"
    "- Start with a COARSE macro plan (usually 3-5 steps). For complex tasks, expand to 6-12 steps.\n"
    "- Prefer graph-first evidence collection before any synthesis.\n"
    "- If you can, DO include explicit tool calls to reduce ambiguity: set `tool` to one of the available tool names.\n"
    "- When you select a tool, you MAY also provide `tool_args` (object) under the step to control tool parameters.\n"
    "  - `tool_args` will be passed into the tool as `extra` (so include keys like focus_query/top_k/max_terms/etc).\n"
    "- Use F-tools first (fast deterministic probes), then X-tools, and use H-tools sparingly.\n"
    "- Use 'graph' for evidence retrieval, 'text' for synthesis/structuring, and 'web' only as a fallback.\n"
    "- Keep each step description grounded with key entities/terms from the question (avoid generic filler like '根据资料…').\n"
    "- The plan may be refined later as evidence arrives.\n"
    "- Never exceed {max_steps}.\n"
    "- Prefer merging trivial lookups into a single step instead of emitting fillers.\n"
    "- By default steps execute sequentially. Only mark a step as parallel-safe when you are confident it is "
    "independent: set `scheduler` to 'parallel' (or `parallelizable`: true) inside the step metadata.\n"
    "\n"
    "Output:\n"
    "- Return ONLY a JSON array.\n"
    "- Each item must include: 'description' (string).\n"
    "- Optional fields: 'channel' (one of ['graph','text','web'], default 'graph'), and 'metadata' (object).\n"
    "- Optional fields: 'tool' (string tool name) and 'tool_args' (object tool parameters).\n"
)
