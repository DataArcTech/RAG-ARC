"""Prompts for hierarchical event schema (HS) extraction.

These prompts are used by HippoRAG2Extractor when SDF extraction is enabled.

Contract (HS - Hierarchical Structure):
- Output MUST contain one or more event blocks separated by a blank line.
- Each block MUST include the following lines (case-sensitive keys):
  - event: <name>  (or subevent: <name>)
  - event_id: <id> (short id, e.g. ev1 / ev1.1 / ev1.1.1)
  - description: <one concise sentence>
  - participants: xxxx | <childName> <childEventId>_P<importance>, ...
  - Gate: xxxx | and | or | xor
  - Relations: xxxx | <eventIdA>><eventIdB>, ...
  - attributes: xxxx | { ... JSON object ... }
"""

HIPPORAG2_SDF_HS_SYSTEM = """You extract a hierarchical event/process schema from a paragraph.
Focus on processes, rules, decisions, and temporal dependencies (what happens, under what conditions, in what order).

Output ONLY the HS blocks in plain text (no explanations, no Markdown code fences).

Formatting rules (MUST follow):
- Separate blocks with exactly one blank line.
- Each block MUST contain these lines:
  event: <name>  (use subevent: for sub-events)
  event_id: <id> (use short ids like ev1, ev1.1, ev1.1.1; keep consistent across blocks)
  description: <one concise sentence>
  participants: xxxx OR "<childName> <childEventId>_P<importance>, ..." where each participant refers to a CHILD EVENT
  Gate: xxxx OR and/or/xor  (how participants/children combine)
  Relations: xxxx OR "<eventIdA>><eventIdB>, ..." for BEFORE dependencies among events/subevents
  attributes: xxxx OR a JSON object with optional keys:
    - temporal: {effective_date?, valid_from?, valid_to?} (ISO-8601 preferred)
    - scope: string
    - conditions: [string]
    - exceptions: [string]
    - priority: number (higher wins) OR string

Constraints:
- Keep names short and reusable (concept-like), but represent the flow (process-like).
- Prefer representing dates/amounts/ratios/terms in attributes rather than as standalone events unless they are actual steps.
"""

HIPPORAG2_SDF_HS_ONE_SHOT_INPUT = """Three main methods are used in lithium-ion recycling: pyrometallurgical, hydrometallurgical, and bioleaching.
Pyrometallurgical employs extreme heat to transform metal oxides into cobalt, copper, iron, and nickel alloys."""

HIPPORAG2_SDF_HS_ONE_SHOT_OUTPUT = """event: lithium-ion recycling
event_id: ev1
description: Recycle lithium-ion batteries via multiple methods.
participants: pyrometallurgical ev1.1_P1, hydrometallurgical ev1.2_P1, bioleaching ev1.3_P1
Gate: or
Relations: xxxx
attributes: xxxx

subevent: pyrometallurgical
event_id: ev1.1
description: Recover metals using extreme heat.
participants: metal oxides ev1.1.1_P1, cobalt alloys ev1.1.2_P0.5
Gate: and
Relations: ev1.1.1>ev1.1.2
attributes: xxxx"""

HIPPORAG2_SDF_HS_PROMPT = """
{system}

Example input:
{example_input}

Example output:
{example_output}

Now extract HS blocks from:
{passage}
"""


HIPPORAG2_SDF_HS_SYSTEM_ZH = """你需要从段落中抽取“层级事件/过程 Schema（HS）”，用于后续转成 SDF（Schema Definition Format）。
重点抽取：流程步骤、规则裁决、条件/例外、以及时序依赖（before）。

只能输出 HS 结构块（纯文本），不要输出解释、不要输出 Markdown 代码块。

格式约束（必须遵守）：
- 每个事件/子事件是一块；块与块之间用一个空行分隔。
- 每个块必须包含以下行（key 大小写与冒号必须一致）：
  event: <名称>（子事件用 subevent:）
  event_id: <id>（短 id，例如 ev1 / ev1.1 / ev1.1.1，保持一致）
  description: <一句话描述>
  participants: xxxx 或 "<子事件名> <子事件id>_P<importance>, ..."（participants 仅用于指代“子事件”）
  Gate: xxxx 或 and/or/xor（子事件组合逻辑）
  Relations: xxxx 或 "<eventIdA>><eventIdB>, ..."（表示 before 时序）
  attributes: xxxx 或 JSON 对象，可选字段：
    - temporal: {effective_date?, valid_from?, valid_to?}（尽量 ISO-8601）
    - scope: string（适用范围）
    - conditions: [string]（适用条件）
    - exceptions: [string]（除外/免赔/不适用）
    - priority: number 或 string（冲突裁决优先级）

约束：
- 名称尽量短且可复用（偏“概念”），但要能表达“过程/规则链路”（偏“裁决流程”）。
- 日期/金额/比例/期限等优先放在 attributes 里，不要无意义拆成独立事件。
"""

HIPPORAG2_SDF_HS_ONE_SHOT_INPUT_ZH = """保险责任判断通常包含：责任范围、除外责任、免赔额与生效时间。
若在生效日前发生事故，则不承担责任；若触发除外条款，则拒赔。"""

HIPPORAG2_SDF_HS_ONE_SHOT_OUTPUT_ZH = """event: 保险责任裁决
event_id: ev1
description: 对给定事故判断是否属于保险责任并给出结论。
participants: 生效期校验 ev1.1_P1, 除外条款校验 ev1.2_P1, 免赔额计算 ev1.3_P0.8
Gate: and
Relations: ev1.1>ev1.2, ev1.2>ev1.3
attributes: {"scope":"通用保险责任裁决","conditions":["事故发生且已投保"],"exceptions":[]}

subevent: 生效期校验
event_id: ev1.1
description: 判断事故时间是否落在保单有效期内。
participants: xxxx
Gate: xxxx
Relations: xxxx
attributes: {"temporal":{"effective_date":"xxxx","valid_from":"xxxx","valid_to":"xxxx"}}"""

HIPPORAG2_SDF_HS_PROMPT_ZH = """
{system}

示例输入：
{example_input}

示例输出：
{example_output}

现在从以下段落抽取 HS：
{passage}
"""

