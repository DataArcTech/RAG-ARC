from typing import Final

MINDMAP_GENERATION_SYSTEM_PROMPT_ZH: Final[str] = (
    "你是一个专业的思维导图生成助手。请根据用户的问题、回答和检索到的文档片段，生成结构化的思维导图数据。"
)

MINDMAP_GENERATION_USER_PROMPT_TEMPLATE_ZH: Final[str] = """请基于以下信息生成思维导图的节点(nodes)和边(edges)数据:

用户问题: {query}

回答内容: {response}

检索到的文档片段:
{chunks_text}

请生成一个JSON格式的思维导图结构,包含以下字段:
1. nodes: 节点数组,每个节点包含:
   - id: 节点唯一标识
   - name: 节点名称
   - category: 节点分类
   - weight: 节点深度(1为根节点,2为二级节点,3为三级节点,以此类推)

2. edges: 边数组,每个边包含:
   - id: 边的唯一标识
   - weight: 边的权重(0-1之间的浮点数)
   - source: 源节点id
   - target: 目标节点id
   - relation: 关系类型(如"包含"、"说明"、"步骤"、"依据"等)

要求:
- 根节点(weight=1)应该是对整个回答的总结
- 二级节点(weight=2)是主要的知识点或步骤
- 三级节点(weight=3)是详细的说明或子步骤
- 边的source应该是父节点,target应该是子节点
- 边的weight建议: 一级到二级为0.85,二级到三级为0.8
- 所有节点的id和name要清晰明确,便于理解
- 对于根节点和二级节点(weight=1和2)的category都是其自己的name，三级节点及更高节点的都和父节点的name相同。

请直接返回JSON格式的数据,不要包含任何其他说明文字。格式如下:
{{
  "nodes": [
    {{"id": "根节点标题", "name": "根节点标题", "category": "根节点标题", "weight": 1}},
    {{"id": "二级节点标题", "name": "二级节点标题", "category": "二级节点标题", "weight": 2}},
    {{"id": "三级节点标题", "name": "三级节点标题", "category": "二级节点标题", "weight": 3}}
  ],
  "edges": [
    {{"id": "edge_1", "source": "根节点标题", "target": "二级节点标题", "relation": "包含", "weight": 0.85}},
    {{"id": "edge_2", "source": "二级节点标题", "target": "三级节点标题", "relation": "说明", "weight": 0.8}}
  ]
}}
"""


def build_mindmap_generation_user_prompt(*, query: str, response: str, chunks_text: str) -> str:
    return MINDMAP_GENERATION_USER_PROMPT_TEMPLATE_ZH.format(
        query=str(query or "").strip(),
        response=str(response or "").strip(),
        chunks_text=str(chunks_text or "").strip(),
    )


MINDMAP_MERGE_SYSTEM_PROMPT_EN: Final[str] = (
    "You are an experienced knowledge engineer. Merge multiple partial mind maps into one coherent global mind map."
)

MINDMAP_MERGE_USER_PROMPT_TEMPLATE_EN: Final[str] = """We are building a global mind map for the document '{filename}' from multiple segments.
Each segment provides a content summary and may also include a partial mind map fragment (TSV). If a segment has no TSV fragment, infer structure from the content summary.

Output requirements:
1) Use numbering like 1, 1.1, 1.1.1 to represent hierarchy; numbering must be consistent.
2) Output TSV only. Each line must be: '<number>\\t<content>\\t<relation_type>'. 
   - For hierarchical parent-child relationships, use "contains"
   - For other semantic relationships (e.g., "主要客户", "参与", "发布", "采用", "指导"), extract from the content
   - If no specific relation is mentioned, use "contains" as default
3) The root node (1) should summarize the entire document at a high level.
4) Lower-level nodes should cover key information concisely and accurately.
5) If there are duplicates or conflicts, reconcile them and keep the structure coherent.

Segments:
{sections_text}

Now output the merged TSV mind map:
"""


def build_mindmap_merge_user_prompt(*, filename: str, sections_text: str) -> str:
    return MINDMAP_MERGE_USER_PROMPT_TEMPLATE_EN.format(
        filename=str(filename or "").strip(),
        sections_text=str(sections_text or "").strip(),
    )
