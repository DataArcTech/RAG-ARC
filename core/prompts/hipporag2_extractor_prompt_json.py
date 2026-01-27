"""
Graph extraction prompts (HippoRAG2 pipeline) using strict JSON outputs.

These prompts are:
- Output must be valid JSON (no TSV).
- Entity references use integer IDs to prevent endpoint drift.
- relation_type uses SCREAMING_SNAKE_CASE.
"""

HIPPORAG2_EDGE_SCHEMA_HINT_EN = """Preferred canonical relation types (SCREAMING_SNAKE_CASE):
{allowed_relations}

Alias map (raw -> canonical):
{relation_aliases_json}
"""

HIPPORAG2_EDGE_SCHEMA_HINT_ZH = """优先使用的关系类型（SCREAMING_SNAKE_CASE）：
{allowed_relations}

别名映射（raw -> canonical）：
{relation_aliases_json}
"""

# =========================
# Entities (NER)
# =========================

HIPPORAG2_NER_JSON_SYSTEM_EN = """You are an expert information extraction model.
Extract disambiguated named entities from the provided passage.

Rules:
- Resolve pronouns and ambiguous references to explicit entity names.
- Exclude vague/unresolved mentions when referents cannot be determined.
- Output MUST be a single JSON object matching the required schema. No markdown, no commentary.
"""

HIPPORAG2_NER_JSON_SYSTEM_ZH = """你是专业的信息抽取模型。
从给定段落中抽取“已消歧”的命名实体。

规则：
- 将代词/指代消解为明确的实体名称。
- 不能确定指代对象时，不要输出该实体。
- 输出必须是“单个 JSON 对象”，且严格符合 schema；不要输出 Markdown、解释或多余文本。
"""

HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_EN = """Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City launched PlanetRadiocity.com in May 2008."""

HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_EN = """{
  "extracted_entities": [
    {"id": 1, "name": "Radio City", "entity_type": "ORGANIZATION"},
    {"id": 2, "name": "India", "entity_type": "LOCATION"},
    {"id": 3, "name": "3 July 2001", "entity_type": "DATE"},
    {"id": 4, "name": "Hindi", "entity_type": "LANGUAGE"},
    {"id": 5, "name": "English", "entity_type": "LANGUAGE"},
    {"id": 6, "name": "May 2008", "entity_type": "DATE"},
    {"id": 7, "name": "PlanetRadiocity.com", "entity_type": "ORGANIZATION"}
  ]
}"""

HIPPORAG2_NER_JSON_ONE_SHOT_INPUT_ZH = """小米公司于 2010 年在北京成立。雷军是小米的创始人之一。"""

HIPPORAG2_NER_JSON_ONE_SHOT_OUTPUT_ZH = """{
  "extracted_entities": [
    {"id": 1, "name": "小米公司", "entity_type": "ORGANIZATION"},
    {"id": 2, "name": "2010年", "entity_type": "DATE"},
    {"id": 3, "name": "北京", "entity_type": "LOCATION"},
    {"id": 4, "name": "雷军", "entity_type": "PERSON"}
  ]
}"""

HIPPORAG2_NER_JSON_PROMPT_EN = """
{system}

Schema:
{{
  "extracted_entities": [
    {{"id": 1, "name": "...", "entity_type": "PERSON|ORGANIZATION|LOCATION|DATE|..."}}
  ]
}}

Example input:
{example_input}

Example output:
{example_output}

Now extract entities from:
{passage}

Output JSON:
""".strip()

HIPPORAG2_NER_JSON_PROMPT_ZH = """
{system}

Schema:
{{
  "extracted_entities": [
    {{"id": 1, "name": "...", "entity_type": "PERSON|ORGANIZATION|LOCATION|DATE|..."}}
  ]
}}

示例输入：
{example_input}

示例输出：
{example_output}

现在请从下文抽取实体：
{passage}

输出 JSON：
""".strip()


# =========================
# Edges (Triples)
# =========================

HIPPORAG2_EDGE_JSON_SYSTEM_EN = """You are an expert fact extractor that extracts factual relationships from text.
Output MUST be a single JSON object matching the required schema. No markdown, no commentary.
"""

HIPPORAG2_EDGE_JSON_SYSTEM_ZH = """你是专业的事实抽取模型，从文本中抽取实体之间的事实关系。
输出必须是“单个 JSON 对象”，且严格符合 schema；不要输出 Markdown、解释或多余文本。
"""

HIPPORAG2_EDGE_JSON_PROMPT_EN = """
{system}

<FACT TYPES>
{edge_types}
</FACT TYPES>

<ENTITIES>
{entities_json}
</ENTITIES>

<REFERENCE_TIME>
{reference_time}
</REFERENCE_TIME>

<PASSAGE>
{passage}
</PASSAGE>

TASK:
Extract factual relationships between the given ENTITIES based on the PASSAGE.

Rules:
- Use ONLY entity ids from ENTITIES for source_entity_id/target_entity_id.
- Each edge must connect two DISTINCT entities.
- Use SCREAMING_SNAKE_CASE for relation_type (e.g., WORKS_AT, LOCATED_IN).
- If no suitable type exists, use RELATES_TO.
- Do not emit duplicates or near-duplicates.
- fact: paraphrase the source sentence(s) (optional but recommended for debugging).
- valid_at/invalid_at: ISO-8601 datetime with Z, or null when unknown.

Schema:
{{
  "edges": [
    {{
      "relation_type": "WORKS_AT",
      "source_entity_id": 1,
      "target_entity_id": 2,
      "fact": "optional",
      "valid_at": "2025-01-01T00:00:00Z",
      "invalid_at": null
    }}
  ]
}}

Output JSON:
""".strip()

HIPPORAG2_EDGE_JSON_PROMPT_ZH = """
{system}

<事实类型（可参考，但不穷举）>
{edge_types}
</事实类型（可参考，但不穷举）>

<实体列表（只能用这些 id）>
{entities_json}
</实体列表（只能用这些 id）>

<参考时间>
{reference_time}
</参考时间>

<正文>
{passage}
</正文>

任务：从正文中抽取实体之间的事实关系（仅限实体列表内的实体）。

规则：
- source_entity_id/target_entity_id 只能使用实体列表中的 id。
- 每条关系必须连接两个不同实体。
- relation_type 必须是 SCREAMING_SNAKE_CASE（例如 WORKS_AT, LOCATED_IN）。
- 如果没有合适类型，用 RELATES_TO。
- 不要输出重复或语义高度重复的事实。
- fact：可选，但建议输出对原句的“近义改写”以便溯源/排错。
- valid_at/invalid_at：ISO-8601（UTC，Z 后缀）；无法确定则为 null。

输出 schema：
{{
  "edges": [
    {{
      "relation_type": "WORKS_AT",
      "source_entity_id": 1,
      "target_entity_id": 2,
      "fact": "可选",
      "valid_at": "2025-01-01T00:00:00Z",
      "invalid_at": null
    }}
  ]
}}

输出 JSON：
""".strip()
