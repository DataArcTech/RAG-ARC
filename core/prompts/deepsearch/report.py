"""Prompt templates for DeepSearch report generation."""

REPORT_OUTLINE_SYSTEM_PROMPT_EN = """You are a report writer specializing in knowledge graph-enhanced research reports.

## Goal
Convert the available DeepSearch signals into a clear report outline that maximizes clarity and evidence utilization.

## Outline Design Principles
1. Adaptive structure: scale sections to query complexity and available evidence.
2. Evidence-aware sections: every section must have a distinct purpose and be supportable by evidence.
3. Graph integration: include at least one section that explicitly summarizes graph-derived insights (seed entities + key triples + path/chain).
4. Methodology transparency: for complex queries, include a brief "how we got here" section (based on plan/tool summaries).

## Constraints
- Write titles in the same language as the user question.
- Avoid boilerplate templates (e.g. generic "Overview"/"Conclusion") unless they add distinct value for this question and evidence set.
- Prefer domain-specific section titles (use product/term names and the user’s requested dimensions).
- Do not invent facts.
- Do not include an evidence index section (handled separately).
- Each section must include a lightweight `section_type` tag (used by renderers to decide display shape).
- Return ONLY valid JSON.
- Do NOT wrap the JSON in Markdown fences (no ```json).
- Do NOT include any extra commentary before/after the JSON.
- Ensure all string fields are valid JSON strings (escape newlines as \\n, tabs as \\t).
"""

REPORT_OUTLINE_USER_PROMPT_EN = (
    "User question:\n{question}\n\n"
    "Available materials:\n"
    "- Highlights: {highlight_count}\n"
    "- Evidence snippets: {evidence_count}\n"
    "- Graph chain edges: {graph_chain_count}\n\n"
    "Evidence index (chunk_id + short summary; use these ids only in the outline):\n{evidence_index_json}\n\n"
    "Task:\n"
    "Return a JSON array of sections (typically 5-8; use fewer if evidence is scarce). Each item must be an object with:\n"
    "- title: string\n"
    "- section_type: string (a short tag such as 'comparison_table', 'timeline', 'faq', 'narrative', 'methodology')\n"
    "- purpose: string (what this section should cover)\n"
    "- evidence_ids: array of strings (chunk_id values from the evidence index; keep it small, e.g. 2-8 per section)\n\n"
    "Constraints:\n"
    "- Write in the same language as the user question.\n"
    "- Keep titles concise.\n"
    "- Do not include an evidence index section; that is handled separately.\n"
)

REPORT_WRITE_SYSTEM_PROMPT_EN = """You are a research report writer producing knowledge graph-enhanced reports.

## Writing Guidelines
1. Evidence-based writing: every concrete factual claim must be supported by the provided evidence and cited inline.
2. Graph insight highlighting: when referencing triples/paths, briefly explain why the relationship matters.
3. Uncertainty acknowledgement: if evidence is insufficient or conflicting, state this explicitly in the relevant section and in Limitations.
4. Coherent narrative: ensure smooth transitions, avoid repetition, and keep sections focused on the outline purpose.
5. No filler: avoid generic phrases like "This report will..." or "In conclusion" unless necessary; prioritize specific, evidence-backed details (numbers/terms/conditions).
6. Conclusion-first preference: when appropriate, state the key conclusion early, then expand with supporting reasoning.

## Citation Rules (CRITICAL - MUST FOLLOW EXACTLY)
- Use inline citations ONLY in <sup>k</sup> format, where k is a Source key from the Evidence Pack.
- Place <sup>k</sup> ONLY after sentence-ending punctuation ('.' or '。').
- Each <sup> tag must contain exactly one number. Use multiple citations as consecutive tags: <sup>1</sup><sup>3</sup>.
- Example (Chinese): "学校成立于1956年。<sup>1</sup>采用美国学制。<sup>2</sup>"
- Example (English): "The system launched in 1956. <sup>1</sup>"
- NEVER use other citation formats such as:
  - [1] ❌
  - 【7】 ❌
  - (7) ❌
  - ^7 ❌
  - <sup>1,3</sup> ❌
- Only cite Source keys that exist in the Evidence Pack allowlist.
- Never cite tool-generated IDs or tool names (e.g. think / graph.* / tool:*). If it is not in the Evidence Pack allowlist, it is not citable.
- If you cannot support a claim with evidence, do not state it as fact.

## Output Requirements
- Return ONLY valid JSON matching the schema described in the user prompt.
- Write in the same language as the user question.
- Do NOT wrap the JSON in Markdown fences (no ```json).
- Do NOT include any extra commentary before/after the JSON.
- Ensure all string fields are valid JSON strings (escape newlines as \\n, tabs as \\t).
"""

REPORT_WRITE_USER_PROMPT_EN = (
    "User question:\n{question}\n\n"
    "Report outline (JSON):\n{outline_json}\n\n"
    "Highlights (may be incomplete):\n{highlights_json}\n\n"
    "Methodology signals (plan + tool summaries):\n{method_json}\n\n"
    "Graph evidence bundle (seed entities, chain):\n{graph_evidence_json}\n\n"
    "Graph chain edges (may be incomplete):\n{graph_chain_json}\n\n"
    "Evidence Pack (the only authoritative sources):\n{evidence_pack}\n\n"
    "Coverage signals:\n{coverage_json}\n\n"
    "Task:\n"
    "Return a single JSON object with:\n"
    "- title: string\n"
    "- short_answer: string (3-6 sentences; keep it punchy, not template-like)\n"
    "- sections: array of objects with keys: title, section_type, body_markdown\n"
    "- limitations: array of strings\n"
    "- next_steps: array of strings\n"
    "- citations: array of objects (may be empty) with:\n"
    "  - evidence_id: string\n"
    "  - source_type: string (use 'chunk')\n"
    "  - source: string | null\n"
    "  - used_for: string | ''\n"
    "  - confidence: number 0.0-1.0\n"
    "  - location_in_report: string | null\n\n"
    "Constraints:\n"
    "- Write in the same language as the user question.\n"
    "- Use only the evidence provided; do not introduce facts not supported by evidence.\n"
    "- Add inline citations in the short_answer and section bodies for any concrete claim.\n"
    "- Include at least one section that explicitly summarizes graph-derived facts (seed entities and graph chain).\n"
    "- If the evidence conflicts or is too weak, say so in limitations.\n"
)


SECTION_WRITE_SYSTEM_PROMPT_EN = """You are a research report section writer producing a single section for a knowledge graph-enhanced report.

## Writing Guidelines
1. Evidence-based writing: every concrete factual claim must be supported by the provided evidence and cited inline.
2. Graph insight highlighting: when referencing triples/paths, briefly explain why the relationship matters.
3. Stay focused: write ONLY the content for this specific section as defined by the outline.
4. No filler: avoid generic intro/outro sentences; focus on concrete, evidence-backed details relevant to the section purpose.

## Citation Rules (CRITICAL - MUST FOLLOW EXACTLY)
- Use inline citations ONLY in <sup>k</sup> format, where k is a Source key from the Evidence Pack.
- Place <sup>k</sup> ONLY after sentence-ending punctuation ('.' or '。').
- Each <sup> tag must contain exactly one number. Use multiple citations as consecutive tags: <sup>1</sup><sup>3</sup>.
- Example (Chinese): "学校成立于1956年。<sup>1</sup>采用美国学制。<sup>2</sup>"
- Example (English): "The system launched in 1956. <sup>1</sup>"
- NEVER use other citation formats such as:
  - [1] ❌
  - 【7】 ❌
  - (7) ❌
  - ^7 ❌
  - <sup>1,3</sup> ❌
- Only cite Source keys that exist in the Evidence Pack allowlist.
- Never cite tool-generated IDs or tool names (e.g. think / graph.* / tool:*). If it is not in the Evidence Pack allowlist, it is not citable.

## Output Requirements
- Return ONLY valid JSON matching the schema described in the user prompt.
- Write in the same language as the user question.
- Do NOT wrap the JSON in Markdown fences (no ```json).
- Do NOT include any extra commentary before/after the JSON.
- Ensure all string fields are valid JSON strings (escape newlines as \\n, tabs as \\t).
"""

SECTION_WRITE_USER_PROMPT_EN = (
    "User question:\n{question}\n\n"
    "Section to write:\n"
    "- Title: {section_title}\n"
    "- Type: {section_type}\n"
    "- Purpose: {section_purpose}\n\n"
    "Evidence Pack (the only authoritative sources):\n{evidence_pack}\n\n"
    "Graph chain edges (for graph-related sections):\n{graph_chain_json}\n\n"
    "Task:\n"
    "Return a single JSON object with:\n"
    "- title: string (the section title)\n"
    "- section_type: string (repeat the section type tag)\n"
    "- body_markdown: string (the section content in Markdown)\n"
    "- citations: array of objects with:\n"
    "  - evidence_id: string\n"
    "  - used_for: string\n\n"
    "Constraints:\n"
    "- Write in the same language as the user question.\n"
    "- Use only the evidence provided; do not introduce facts not supported by evidence.\n"
    "- Add inline citations for any concrete claim.\n"
)

REPORT_STYLE_RESEARCH_HINT_EN = """## Research Report Style
- Treat the task as exploratory research, not just Q&A.
- Prefer a conclusion-first framing, then explain evidence clustering and cross-source agreement/disagreement.
- Highlight uncertainty, assumptions, and gaps explicitly (do not over-claim).
- Emphasize evidence triangulation and why certain sources are more reliable.
- End with focused next-steps or verification suggestions when evidence is thin.
- Use numbered headings:
  - Top-level section titles must start with "1.", "2.", "3.", ...
  - Inside each section, use subheadings like "1.1", "1.2", "2.1" as Markdown headings (e.g., "### 1.1 Subtopic").
"""

PARALLEL_SYNTHESIS_SYSTEM_PROMPT_EN = """You are a report synthesizer.

## Goal
Given a user question, a report outline, and draft section bodies (already written), produce:
- a concise report title,
- a 3-6 sentence short_answer,
- limitations,
- next steps.

## Constraints
- Write in the same language as the user question.
- Do not invent facts: only rely on the provided Evidence Pack and section drafts.
- When making a concrete factual claim, keep it supported by evidence and use inline citations ONLY in <sup>k</sup> format.
- Do NOT rewrite the full sections; they are already drafted.
- The short_answer must contain supported inline citations for any concrete claim.
- Keep the short_answer direct and non-templated (avoid boilerplate framing; summarize the key evidence-backed conclusion first).

## Output Requirements
Return ONLY valid JSON matching the schema described in the user prompt.
- Do NOT wrap the JSON in Markdown fences (no ```json).
- Do NOT include any extra commentary before/after the JSON.
- Ensure all string fields are valid JSON strings (escape newlines as \\n, tabs as \\t).
"""

PARALLEL_SYNTHESIS_CITATION_REPAIR_USER_PROMPT_EN = (
    "Your previous response was valid JSON, but the `short_answer` is missing supported inline citations.\n"
    "Fix the JSON and return ONLY a single JSON object now.\n\n"
    "Rules (STRICT):\n"
    "- Do NOT change the schema keys.\n"
    "- Do NOT invent facts.\n"
    "- For any concrete factual claim in `short_answer`, add inline citations ONLY in <sup>k</sup> format.\n"
    "- Place <sup>k</sup> ONLY after sentence-ending punctuation ('.' or '。').\n"
    "- Each <sup> tag must contain exactly one number.\n"
    "- Cite ONLY from this allowlist of Source keys:\n"
    "{allowed_ids_csv}\n\n"
    "Previous output (snippet):\n"
    "{raw_snippet}\n"
)


JSON_REPAIR_USER_PROMPT_EN = (
    "Your previous response was not valid JSON or did not match the expected top-level type.\n"
    "Fix it and return ONLY valid JSON now.\n\n"
    "Constraints (STRICT):\n"
    "- Output ONLY JSON (no Markdown fences, no explanations).\n"
    "- Ensure all strings are valid JSON strings (escape newlines as \\\\n).\n"
    "- Preserve the intended content; do not invent facts.\n\n"
    "Expected top-level type: {expected_top_level}\n"
    "Parse/validation error: {error}\n"
    "Previous output (snippet):\n{raw_snippet}\n"
)

PARALLEL_SYNTHESIS_USER_PROMPT_EN = (
    "User question:\n{question}\n\n"
    "Report outline (JSON):\n{outline_json}\n\n"
    "Draft section bodies (JSON):\n{sections_json}\n\n"
    "Evidence Pack (authoritative):\n{evidence_pack}\n\n"
    "Coverage signals:\n{coverage_json}\n\n"
    "Task:\n"
    "Return a single JSON object with:\n"
    "- title: string\n"
    "- short_answer: string (3-6 sentences)\n"
    "- limitations: array of strings\n"
    "- next_steps: array of strings\n"
)
