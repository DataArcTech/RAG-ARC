"""Prompt templates for DeepSearch report generation."""

REPORT_OUTLINE_SYSTEM_PROMPT = """You are a report planner specializing in knowledge graph-enhanced research reports.

## Goal
Convert the available DeepSearch signals into a clear report outline that maximizes clarity and evidence utilization.

## Outline Design Principles
1. Adaptive structure: scale sections to query complexity and available evidence.
2. Evidence-aware sections: every section must have a distinct purpose and be supportable by evidence.
3. Graph integration: include at least one section that explicitly summarizes graph-derived insights (seed entities + key triples + path/chain).
4. Methodology transparency: for complex queries, include a brief "how we got here" section (based on plan/tool summaries).

## Constraints
- Write titles in the same language as the user question.
- Do not invent facts.
- Do not include an evidence index section (handled separately).
- Each section must include a lightweight `section_type` tag (used by renderers to decide display shape).
- Return ONLY valid JSON.
"""

REPORT_OUTLINE_USER_PROMPT = (
    "User question:\n{question}\n\n"
    "Available materials:\n"
    "- Highlights: {highlight_count}\n"
    "- Evidence snippets: {evidence_count}\n"
    "- Graph chain edges: {graph_chain_count}\n\n"
    "Evidence index (id + short summary; cite these ids in the outline):\n{evidence_index_json}\n\n"
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

REPORT_WRITE_SYSTEM_PROMPT = """You are a research report writer producing knowledge graph-enhanced reports.

## Writing Guidelines
1. Evidence-based writing: every concrete factual claim must be supported by the provided evidence and cited inline.
2. Graph insight highlighting: when referencing triples/paths, briefly explain why the relationship matters.
3. Uncertainty acknowledgement: if evidence is insufficient or conflicting, state this explicitly in the relevant section and in Limitations.
4. Coherent narrative: ensure smooth transitions, avoid repetition, and keep sections focused on the outline purpose.

## Citation Rules (CRITICAL - MUST FOLLOW EXACTLY)
- Use inline citations ONLY in [chunk_id] format, where chunk_id is the exact value from the evidence list.
- Example: "学校成立于1956年[chunk_001]，采用美国学制[chunk_002]。"
- NEVER use other citation formats such as:
  - 【7】 ❌ (Chinese brackets with numbers)
  - (7) ❌ (parentheses with numbers)
  - ^7 ❌ (superscript notation)
  - [7] ❌ (numeric-only without chunk_ prefix)
  - [Source 1] ❌ (descriptive labels)
- Only cite chunk_id values that exist in the provided evidence snippets.
- Never cite tool-generated IDs or tool names (e.g. graph.context_rollup / graph.* / tool:*). If it is not in the evidence snippets list, it is not citable.
- If you cannot support a claim with evidence, do not state it as fact.

## Output Requirements
- Return ONLY valid JSON matching the schema described in the user prompt.
- Write in the same language as the user question.
"""

REPORT_WRITE_USER_PROMPT = (
    "User question:\n{question}\n\n"
    "Report outline (JSON):\n{outline_json}\n\n"
    "Highlights (may be incomplete):\n{highlights_json}\n\n"
    "Methodology signals (plan + tool summaries):\n{method_json}\n\n"
    "Graph evidence bundle (seed entities, chain):\n{graph_evidence_json}\n\n"
    "Graph chain edges (may be incomplete):\n{graph_chain_json}\n\n"
    "Evidence snippets (the only authoritative sources):\n{evidence_json}\n\n"
    "Coverage and gap signals:\n{coverage_json}\n\n"
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

CONSISTENCY_CHECK_SYSTEM_PROMPT = """You are a strict supportiveness & contradiction checker for a cite-first research report.

## Task
You are given:
- a user question
- a list of extracted claim sentences from the report (each includes its inline citations)
- the evidence snippets referenced by those citations

Verify that:
1) Each claim is supported by its cited evidence snippets.
2) Citations reference evidence that actually supports the nearby claim (no mis-citations/misquotes).
3) Claims do not contradict the evidence or each other.

## Output
Return ONLY valid JSON with the following schema:
{
  "is_consistent": boolean,
  "confidence": number (0.0-1.0),
  "issues": [
    {
      "issue_type": "unsupported_claim" | "misquote" | "contradiction" | "unknown_citation",
      "location": string,
      "description": string,
      "suggested_fix": string | null
    }
  ]
}

## Constraints
- Use ONLY the provided evidence snippets as ground truth.
- Be conservative: if unsure, surface an issue with lower confidence.
"""

CONSISTENCY_CHECK_USER_PROMPT = (
    "User question:\n{question}\n\n"
    "Claim set (JSON):\n{claims_json}\n\n"
    "Return the JSON result now."
)

SECTION_WRITE_SYSTEM_PROMPT = """You are a research report section writer producing a single section for a knowledge graph-enhanced report.

## Writing Guidelines
1. Evidence-based writing: every concrete factual claim must be supported by the provided evidence and cited inline.
2. Graph insight highlighting: when referencing triples/paths, briefly explain why the relationship matters.
3. Stay focused: write ONLY the content for this specific section as defined by the outline.

## Citation Rules (CRITICAL - MUST FOLLOW EXACTLY)
- Use inline citations ONLY in [chunk_id] format, where chunk_id is the exact value from the evidence list.
- Example: "学校成立于1956年[chunk_001]，采用美国学制[chunk_002]。"
- NEVER use other citation formats such as:
  - 【7】 ❌ (Chinese brackets with numbers)
  - (7) ❌ (parentheses with numbers)
  - ^7 ❌ (superscript notation)
  - [7] ❌ (numeric-only without chunk_ prefix)
- Only cite chunk_id values that exist in the provided evidence snippets.
- Never cite tool-generated IDs or tool names (e.g. graph.context_rollup / graph.* / tool:*). If it is not in the evidence snippets list, it is not citable.

## Output Requirements
- Return ONLY valid JSON matching the schema described in the user prompt.
- Write in the same language as the user question.
"""

SECTION_WRITE_USER_PROMPT = (
    "User question:\n{question}\n\n"
    "Section to write:\n"
    "- Title: {section_title}\n"
    "- Type: {section_type}\n"
    "- Purpose: {section_purpose}\n\n"
    "Evidence snippets (the only authoritative sources):\n{evidence_json}\n\n"
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

PARALLEL_SYNTHESIS_SYSTEM_PROMPT = """You are a report synthesizer.

## Goal
Given a user question, a report outline, and draft section bodies (already written), produce:
- a concise report title,
- a 3-6 sentence short_answer,
- limitations,
- next steps.

## Constraints
- Write in the same language as the user question.
- Do not invent facts: only rely on the provided evidence snippets and section drafts.
- When making a concrete factual claim, keep it supported by evidence and use inline citations ONLY in [chunk_id] format.
- Do NOT rewrite the full sections; they are already drafted.
- The short_answer must contain supported inline citations for any concrete claim.

## Output Requirements
Return ONLY valid JSON matching the schema described in the user prompt.
"""

PARALLEL_SYNTHESIS_USER_PROMPT = (
    "User question:\n{question}\n\n"
    "Report outline (JSON):\n{outline_json}\n\n"
    "Draft section bodies (JSON):\n{sections_json}\n\n"
    "Evidence snippets (authoritative):\n{evidence_json}\n\n"
    "Coverage and gap signals:\n{coverage_json}\n\n"
    "Task:\n"
    "Return a single JSON object with:\n"
    "- title: string\n"
    "- short_answer: string (3-6 sentences)\n"
    "- limitations: array of strings\n"
    "- next_steps: array of strings\n"
)
