"""Defaults for DeepSearch initial-think plan templates.

These settings control the optional "template mode" for initial planning:
- A light LLM selects a template + fills slots.
- The runtime builds an initial ThinkNote + plan from the template to seed the tool loop.

Design goals:
- Avoid hardcoding heuristics in code: behavior is configured here + in template definitions.
- Keep selection conservative: if the selector output is invalid, fall back to the normal heavy think.
"""

DEFAULT_INITIAL_THINK_TEMPLATE_ENABLED: bool = True

# If None/empty, the selector will use the connector's default model.
# In most deployments, this should be a low-cost model (e.g. configured via llm_connector.low_cost_model_name).
DEFAULT_INITIAL_THINK_TEMPLATE_MODEL_NAME: str | None = None

DEFAULT_INITIAL_THINK_TEMPLATE_TEMPERATURE: float = 0.0
DEFAULT_INITIAL_THINK_TEMPLATE_MAX_TOKENS: int = 800
DEFAULT_INITIAL_THINK_TEMPLATE_JSON_ATTEMPTS: int = 2

