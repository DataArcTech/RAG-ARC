"""Shared prompt snippets for structured JSON outputs (DeepSearch).

Centralize these strings to avoid scattering ad-hoc JSON repair instructions across tools.
"""

# Generic JSON retry instruction used when a tool requires strict JSON output.
JSON_RETRY_INSTRUCTION_EN = (
    "Fix the previous output and return ONLY a single valid JSON object/array.\n"
    "- No markdown fences (no ```json).\n"
    "- No extra prose before/after the JSON.\n"
    "- Keep it compact (short strings, minimal whitespace).\n"
    "- Ensure all quotes/brackets/braces are closed.\n"
)
