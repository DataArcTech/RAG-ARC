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
    "- Start with a COARSE macro plan (usually 3-5 steps) and keep it simple.\n"
    "- Avoid low-level tool choreography in this stage; tool selection happens during execution.\n"
    "- Use 'graph' for evidence retrieval, 'text' for synthesis/structuring, and 'web' only as a fallback.\n"
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
    "- DO NOT include 'tool' unless you have a very strong reason (it will be ignored if invalid).\n"
)
