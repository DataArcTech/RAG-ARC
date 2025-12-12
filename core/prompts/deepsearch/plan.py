"""Prompt templates dedicated to DeepSearch planning."""

GRAPH_PLANNER_SYSTEM_PROMPT = (
    "You are a planning assistant for graph-centric DeepSearch pipelines. "
    "Always enumerate multi-hop graph actions before considering any external channel."
)

GRAPH_PLANNER_USER_PROMPT = (
    "Question: {question}\n"
    "Mode: {mode}\n"
    "Available tools:\n{available_tools}\n"
    "Instructions: Decompose the problem into several steps (usually 3-6 for simple questions) "
    "and never exceed {max_steps}. Prefer merging trivial lookups into a single step instead of emitting fillers. "
    "Return a JSON array where each item has 'description', 'channel', and when using the graph channel also "
    "a 'tool' (plus optional 'tool_profile'). The 'tool' value must match one of the catalog names exactly. "
    "Channel must be one of ['graph','text','web'] and defaults to 'graph'."
)
