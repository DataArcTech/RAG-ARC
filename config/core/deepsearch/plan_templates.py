"""Plan templates for DeepSearch initial think (domain-agnostic).

Templates are meant to reduce initial planning latency by reusing a small set of
high-quality plans and using a light LLM to pick + fill slots.

Important:
- Templates are *navigation* guidance. Citeable evidence must come from read.pages / graph triples.
- Do not encode product-specific rules here; keep templates generic.
"""
from typing import Any, Dict, List


PLAN_TEMPLATES_VERSION: int = 1

# A small starter set. Keep them broad and tool-oriented.
PLAN_TEMPLATES: List[Dict[str, Any]] = [
    {
        "template_id": "doc.fact_lookup.v1",
        "label": "Locate and answer a factual question in a user's document",
        "description": (
            "Use file routing + in-file navigation (toc/tree/section.select) to find the right pages, "
            "then read full pages as evidence and answer."
        ),
        "slots": {
            "topic": "The main subject or noun phrase the user is asking about (short).",
            "attribute": "What to find/verify (e.g., definition, conditions, scope, rules).",
        },
        "plan_items": [
            "Route to the most relevant file for: {topic}. Start with search.file using the full user question.",
            "Within the selected file, navigate sections via toc.tree / tree.* and prefer section.select to shortlist relevant sections.",
            "Read full evidence with read.pages for the best-matching section(s); expand to adjacent pages if a table spans pages.",
            "Answer ONLY using evidence from read.pages and/or graph triples; if evidence is insufficient, read more pages or ask a clarification question.",
        ],
        "initial_tool_calls": [
            {
                "tool_name": "explore",
                "tool_args": {"actions": [{"tool": "search.file", "args": {"query": "{question}", "top_k": 6}}]},
                "rationale": "Route to the most relevant file(s) first (search.file via explore).",
                "parallelizable": False,
            }
        ],
        "report_style": "deepsearch",
    },
    {
        "template_id": "doc.compare.v1",
        "label": "Compare two or more items described in documents (or within one document)",
        "description": (
            "Identify the comparison targets and axis, route to the best file(s), "
            "then read the relevant pages for each target before synthesizing."
        ),
        "slots": {
            "targets": "The comparison targets (short, comma-separated).",
            "axis": "The comparison axis/criteria (short).",
        },
        "plan_items": [
            "Extract comparison targets: {targets}; axis: {axis}.",
            "Use search.file to locate candidate files for each target; start with the single best file first unless the user explicitly asks cross-file comparison.",
            "Use toc.tree / section.select to locate pages covering {axis} for each target, then read full pages via read.pages.",
            "Synthesize a comparison table; cite only from read.pages/graph triples and call out any missing evidence explicitly.",
        ],
        "initial_tool_calls": [
            {
                "tool_name": "explore",
                "tool_args": {"actions": [{"tool": "search.file", "args": {"query": "{question}", "top_k": 8}}]},
                "rationale": "Locate candidate file(s) for the comparison (search.file via explore).",
                "parallelizable": False,
            }
        ],
        "report_style": "deepsearch",
    },
    {
        "template_id": "doc.numeric_table.v1",
        "label": "Find and interpret numeric/table values in a long document",
        "description": (
            "Numeric answers often live in tables that span multiple pages. "
            "Read the full table pages and adjacent notes to avoid truncated-snippet errors."
        ),
        "slots": {
            "metric": "The numeric metric/value being asked (e.g., rate, return, ratio).",
            "context": "Any constraint such as term/plan/version (short).",
        },
        "plan_items": [
            "Restate the metric to locate: {metric}. Constraints/context: {context}.",
            "Route to the right file via search.file; then use section.select to find the table/section most likely containing {metric}.",
            "Read full pages with read.pages and expand to continuous adjacent pages until the whole table + footnotes are covered.",
            "Extract the numeric values carefully; if ambiguous, quote the relevant rows/notes and explain the interpretation boundaries.",
        ],
        "initial_tool_calls": [
            {
                "tool_name": "explore",
                "tool_args": {"actions": [{"tool": "search.file", "args": {"query": "{question}", "top_k": 6}}]},
                "rationale": "Find the file likely containing the target table/value (search.file via explore).",
                "parallelizable": False,
            }
        ],
        "report_style": "deepsearch",
    },
    {
        "template_id": "general.encyclopedia.v1",
        "label": "Encyclopedia-style question not requiring user's files",
        "description": (
            "When the question is a general definition/concept question, do not force file grounding. "
            "Answer briefly and ask whether file-grounded analysis is desired."
        ),
        "slots": {
            "concept": "The concept being asked (short).",
        },
        "plan_items": [
            "Answer the general concept question about: {concept}.",
            "Ask whether the user wants a file-grounded DeepSearch (and what file/product/period to use).",
        ],
        "initial_tool_calls": [],
        "report_style": "deepsearch",
    },
]
