"""Prompt templates dedicated to DeepSearch planning."""

GRAPH_PLANNER_SYSTEM_PROMPT = (
    "You are a planning assistant for graph-centric DeepSearch pipelines. "
    "Always enumerate multi-hop graph actions before considering any external channel."
)

GRAPH_PLANNER_USER_PROMPT = (
    "Question: {question}\n"
    "Mode: {mode}\n"
    "Available tools:\n{available_tools}\n"
    "Instructions: Decompose the problem into at most {max_steps} steps. "
    "Return a JSON array where each item has 'description' and 'channel'. "
    "Channel must be one of ['graph','text','web'] and defaults to 'graph'."
)
