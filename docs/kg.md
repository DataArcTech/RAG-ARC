# 知识图谱替代/增强 RAG 的生产指南（严谨版）

## 1. 文档目标与范围

**目标**：用可验证、可解释、可工程化的方式，系统性解决标准 RAG（向量检索 + Top‑K 上下文 + LLM 生成）在生产中常见的 12 类故障模式；给出可落地的数据建模、检索与推理方法，并保留可复现实验的示例场景与示例代码。

**范围**：

- 以“知识图谱（Knowledge Graph, KG）”作为**事实存储与约束层**，用于：
  - 可控检索（过滤/路由/子图召回）
  - 事实校验（边验证/方向性/消歧）
  - 计算与规则（时间、集合、计数、逻辑规则）
- 不讨论特定图数据库（Neo4j/JanusGraph/AGE 等）的选型细节；示例代码以 Python + NetworkX 的属性图做演示（生产可替换为图数据库）。

## 2. 背景：标准 RAG 的结构性上限

标准 RAG 的核心依赖是 **embedding 相似度** 与 **Top‑K chunk 召回**。这带来三类结构性限制：

1. **相似度不等于事实**：embedding 对“共现”和“相似语义”敏感，但不保证关系真实性与方向性。
2. **Top‑K 有物理上限**：chunking 会打碎结构；Top‑K 与上下文长度共同导致“碎片化（Fragmentation）”与“召回不全（Recall Ceiling）”。
3. **LLM 不是数据库/规则引擎**：计数、去重、集合差、冲突裁决等需要确定性计算，而非概率生成。

KG 的工程价值在于：把“概率生成层”与“确定性事实/计算层”解耦，使系统具备可解释、可测试、可审计的行为边界。

## 3. 12 类故障模式总览（需求 → 方法）

| 编号 | 故障模式（痛点需求） | RAG 典型失败形态 | KG 处理方法（解法） |
|---:|---|---|---|
| 01 | 多跳推理 | 上下文缺少链路，无法跨片段推导 | 图遍历/路径查询 |
| 02 | 实体歧义 | 同名实体混入上下文，答案跑偏 | 实体链接 + 类型化路由/过滤 |
| 03 | 隐含关系幻觉 | 共现即“被推断为有关系” | 显式边验证 + Schema 约束 |
| 04 | 方向性反转 | 主谓/债权/股权方向被颠倒 | 有向边 + 语义角色标注 |
| 05 | 零散证据 | Top‑K 导致关键条款漏召回 | 邻域召回（中心辐射） |
| 06 | 结构层级 | 只能看到局部父节点，丢失祖先链 | 递归遍历（寻根/溯源） |
| 07 | 交叉点（共同邻居） | 无法合成隐含交集关系 | 交集查询 + 规则判定 |
| 08 | 同义词/行话 | “黑话”无法命中关键日志/文档 | 本体映射 + 查询扩展 |
| 09 | 矛盾事实（时间冲突） | 新旧政策混答或引用旧规 | 时间属性 + 最新真值裁决 |
| 10 | 因果综合（规则） | 无法合成“组合风险” | 规则引擎/逻辑检查 |
| 11 | 否定与缺失 | “不含 X”无法检索到“不提 X”的安全项 | 集合差/补集计算 |
| 12 | 聚合（计数/统计） | 数错、漏数、重复计数 | 实体归一化 + 聚合计数 |

> 本仓库落地：上述 01–12 的“可计算问题”对应的确定性图工具位于 `core/deepsearch/tools/fast/graph_ops_*.py`，
> 回归集（最小可复现样例）位于 `test/deepsearch/test_graph_deterministic_tools.py`（命名 `test_example_01`…`test_example_12`）。

> 工具证据语义（严谨约束）：本仓库在 `ToolDescriptor.strategy_tags` 中区分
> - `evidence_primary`：可引用证据（要求 provenance 可追溯到原始 chunk 或 `fact_id/source_chunk_ids`）
> - `evidence_derived`：派生中间产物（总结/推导/缓存/候选路径等），**不可作为引用证据**
>
> 对齐治理回归：`uv run pytest -q test/deepsearch/test_tool_descriptor_governance.py`。

## 4. 示例实现的统一前置（可替换为生产组件）

> 说明：原始示例来自三篇文章的实验代码，本文将其组织为更可复用的结构。示例使用 `llm.invoke(prompt)` 作为抽取器接口；生产中建议替换为可版本化的抽取模型/模板，并将 Schema 与提示词纳入统一配置管理。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import networkx as nx

try:
    from langchain_core.documents import Document
except Exception:  # 示例允许缺省依赖，生产中应固定依赖版本
    @dataclass
    class Document:
        page_content: str


def parse_triplet(line: str) -> List[str]:
    parts = [p.strip() for p in line.split("|")]
    return [p for p in parts if p]
```

---

## 01 多跳推理（Graph Traversal）

### 痛点需求

已知分散事实：`A 控股 B`、`B 控股 C`，系统需要对问题 `A 是否间接控股 C` 给出确定性回答。

### 处理方法

- 数据建模：将“控股”建模为有向边 `(A)-[:OWNS]->(B)`。
- 查询执行：对图执行路径查询（BFS/DFS），而不是依赖 Top‑K chunk 拼接。

### 示例例子

股权链：`A -> B -> C`；查询：`A 是不是 C 的老板？`

### 示例代码

```python
import networkx as nx

kg = nx.DiGraph()
kg.add_edge("A公司", "B公司", relation="OWNS")
kg.add_edge("B公司", "C公司", relation="OWNS")

def owns_transitively(graph: nx.DiGraph, a: str, c: str) -> bool:
    return nx.has_path(graph, a, c)

assert owns_transitively(kg, "A公司", "C公司") is True
```

---

## 02 实体歧义（Entity Disambiguation）

### 痛点需求

用户查询“捷豹/苹果”时，系统必须区分同名实体（公司/动物/软件），避免检索与生成被“同词不同义”污染。

### 处理方法

- 建图阶段：实体链接（Entity Linking），为节点分配类型标签（如 `Company/Animal/Software`）。
- 查询阶段：先做意图识别/分类（可用轻量分类器或规则），再**按类型过滤**进行子图检索与上下文构造。

### 示例例子

查询：`捷豹在第三季度表现如何？`  
上下文包含三类“捷豹”：公司新闻、动物行为、macOS 版本。

### 示例代码（保留原示例的核心逻辑）

```python
# --- 模拟存在歧义的数据 ---
raw_texts = [
    "捷豹路虎因SUV销量强劲，在第三季度营收增长了12%。",  # 公司
    "受有利天气影响，亚马逊地区的美洲豹种群在第三季度展现出较高的捕猎效率。",  # 动物
    "捷豹（macOS 10.2）系统性能基准测试在最新的第三季度补丁中显著提升。"  # 软件
]
docs = [Document(page_content=t) for t in raw_texts]

# --- 解决方案：基于意图的图谱过滤（示意） ---
kg = nx.DiGraph()
kg.add_node("捷豹路虎(公司)", type="Company")
kg.add_node("美洲豹(动物)", type="Animal")
kg.add_node("捷豹(macOS)", type="Software")
kg.add_edge("捷豹路虎(公司)", "Q3营收增长12%", relation="HAS_FACT")
kg.add_edge("美洲豹(动物)", "Q3捕猎效率更高", relation="HAS_FACT")
kg.add_edge("捷豹(macOS)", "Q3补丁提升性能", relation="HAS_FACT")

def resolve_query_intent(query: str) -> str:
    # 生产中应为可测试的分类器；此处对齐原示例：模拟“公司”意图
    return "Company"

query = "捷豹在第三季度表现如何？"
intent_type = resolve_query_intent(query)

relevant_facts: list[str] = []
for node, attrs in kg.nodes(data=True):
    if attrs.get("type") == intent_type:
        for neighbor in kg.successors(node):
            relevant_facts.append(f"{node} {kg[node][neighbor]['relation']} {neighbor}")

filtered_context = "；".join(relevant_facts)
assert "Q3营收增长12%" in filtered_context
```

---

## 03 隐含关系幻觉（Co‑occurrence Hallucination）

### 痛点需求

实体 A 与 B 在同一段落共现（峰会/名单/摘要）时，系统必须避免“共现→合作”的错误推断，回答应以**已验证关系**为准。

### 处理方法

- 关系抽取引入 **Schema（允许关系集）**：只允许业务定义过的关系进入图谱。
- 查询时做 **显式边验证**：例如仅当存在 `PARTNERS_WITH` 边，才允许回答“有合作”。

### 示例例子

查询：`特斯拉是否与丰田有电池合作关系？`  
陷阱：峰会新闻同时提及“特斯拉、丰田、电池”，但并未陈述合作事实。

### 示例代码（保留原示例的核心逻辑）

```python
ALLOWED_RELATIONS = {"PARTNERS_WITH", "SUPPLIES", "DEVELOPS", "ACQUIRED"}
kg = nx.DiGraph()

def accept_edge(subj: str, rel: str, obj: str) -> None:
    rel_norm = rel.upper().replace(" ", "_")
    if rel_norm in ALLOWED_RELATIONS:
        kg.add_edge(subj, obj, relation=rel_norm)

accept_edge("松下", "SUPPLIES", "特斯拉")
accept_edge("特斯拉", "DISCUSSED_WITH", "丰田")  # 会被拒绝

def verify_partnership(a: str, b: str) -> bool:
    return kg.has_edge(a, b) and kg[a][b]["relation"] == "PARTNERS_WITH"

assert verify_partnership("特斯拉", "丰田") is False
```

---

## 04 方向性反转（Directionality）

### 痛点需求

在所有权/债权/调用链等场景中，系统必须严格区分方向（`A 拥有 B` ≠ `B 拥有 A`），避免主谓颠倒。

### 处理方法

- 将关系建模为有向边，并在抽取阶段做**语义角色标注**（统一归一化到 `母公司 | OWNS | 子公司` 等规范关系）。
- 查询阶段分别支持“上游（入边）”与“下游（出边）”分析。

### 示例例子

查询：`谁拥有 Stratos Global？`  
事实：`Novacorp -> Stratos Global`（隶属于…旗下）。

### 示例代码（保留原示例的核心逻辑）

```python
kg = nx.DiGraph()
kg.add_edge("Novacorp", "Stratos Global", relation="OWNS")
kg.add_edge("Stratos Global", "TinyAI", relation="OWNS")  # 收购行为

target = "Stratos Global"
parents = list(kg.predecessors(target))   # 谁拥有 target
children = list(kg.successors(target))    # target 拥有谁

assert parents == ["Novacorp"]
assert "TinyAI" in children
```

---

## 05 零散证据（Fragmented Evidence / Recall Ceiling）

### 痛点需求

当一个实体的关键属性分散在多页/多文档中（如合同条款、产品认证、配置项），系统必须做到“无限 k 的完整召回”，避免 Top‑K 漏项。

### 处理方法

- 建图阶段：将“主体实体”作为中心节点，把所有属性/条款作为邻接节点聚合。
- 查询阶段：对中心节点做邻域召回（1‑hop 或 k‑hop），以图结构替代 Top‑K chunk 的物理上限。

### 示例例子

场景：`Model‑X` 的安全认证与特性分散在多页；RAG 只能返回部分页面。

### 示例代码（保留原示例的核心逻辑）

```python
kg = nx.DiGraph()
for feature in ["钛合金机身", "浪涌保护", "激光雷达避障", "红色急停按钮", "ISO-9001 认证"]:
    kg.add_edge("Model-X", feature, relation="HAS_FEATURE")

def list_all_features(product_id: str) -> list[str]:
    return list(kg.successors(product_id))

features = list_all_features("Model-X")
assert len(features) == 5
```

---

## 06 结构层级（Hierarchy / Lineage）

### 痛点需求

当层级关系分散在多段文本（BOM、系统分解、组织架构）中，系统需要从叶子节点（部件）追溯到根节点（系统），输出完整链路。

### 处理方法

- 建图：抽取并保存 `CONTAINS`（父级包含子级）关系。
- 查询：从叶子节点逆向递归遍历前驱节点，直到根节点（无前驱）。

### 示例例子

`Zeus‑X` 包含 `核心处理单元`；其包含 `量子芯片组`；其包含 `Qubit Lattice`。查询要追溯 `Qubit Lattice` 的完整层级路径。

### 示例代码（补全递归遍历，便于工程复现）

```python
kg = nx.DiGraph()
kg.add_edge("Zeus-X 超级计算机", "核心处理单元", relation="CONTAINS")
kg.add_edge("核心处理单元", "量子芯片组", relation="CONTAINS")
kg.add_edge("量子芯片组", "Qubit Lattice", relation="CONTAINS")

def trace_to_root(graph: nx.DiGraph, leaf: str) -> list[str]:
    chain = [leaf]
    cur = leaf
    while True:
        parents = list(graph.predecessors(cur))
        if not parents:
            break
        cur = parents[0]  # 演示：假设单父；生产需处理多父/分支
        chain.append(cur)
    return list(reversed(chain))

assert trace_to_root(kg, "Qubit Lattice") == [
    "Zeus-X 超级计算机",
    "核心处理单元",
    "量子芯片组",
    "Qubit Lattice",
]
```

---

## 07 交叉点（共同邻居 / Intersection Query）

### 痛点需求

当风险由“隐含交集 + 规则”决定（如药物相互作用：A 抑制 X 且 B 由 X 代谢），系统必须做**交集查询**与**规则判定**，避免被无关“共现文档”带偏。

### 处理方法

- 建图：药物与靶点之间的关系（`INHIBITS` / `METABOLIZED_BY`）。
- 查询：计算共同邻居；对共同邻居应用规则。

### 示例例子

`Zenthorax INHIBITS CYP3A4`，`Vira‑X METABOLIZED_BY CYP3A4` → DDI 风险。

### 示例代码（保留原示例的核心逻辑）

```python
kg = nx.DiGraph()
kg.add_edge("Zenthorax", "CYP3A4", relation="INHIBITS")
kg.add_edge("Vira-X", "CYP3A4", relation="METABOLIZED_BY")

def check_ddi(a: str, b: str) -> bool:
    inter = set(kg.successors(a)).intersection(set(kg.successors(b)))
    for target in inter:
        rel_a = kg[a][target]["relation"]
        rel_b = kg[b][target]["relation"]
        if (rel_a, rel_b) in {("INHIBITS", "METABOLIZED_BY"), ("METABOLIZED_BY", "INHIBITS")}:
            return True
    return False

assert check_ddi("Zenthorax", "Vira-X") is True
```

---

## 08 同义词与行话（Ontology Mapping / Query Expansion）

### 痛点需求

组织内部“黑话/代号/技术标识符”与用户使用的业务术语不一致时，系统必须能把查询映射到正确的技术实体与日志片段。

### 处理方法

- 将服务目录/CMDB 作为权威来源载入图谱：`技术ID --[IMPLEMENTS]--> 业务服务`。
- 查询阶段：从业务服务反查技术ID，做查询扩展与检索过滤。

### 示例例子

用户问“结账服务错误率”；日志写的是 `Cart-Flow-V2 500 Error Rate 15%`。

### 示例代码（保留原示例的核心逻辑）

```python
kg = nx.DiGraph()
kg.add_edge("Cart-Flow-V2", "结账服务", relation="IMPLEMENTS")

docs = [
    Document(page_content="[System Log] Service: Cart-Flow-V2 | Status: CRITICAL | Metric: 500 Error Rate is at 15% due to DB timeout."),
    Document(page_content="[System Log] Service: Stripe-Adaptor | Status: HEALTHY | Metric: Latency < 20ms."),
    Document(page_content="[Dev Team Chat] The Checkout UI team is deploying a new CSS fix for the button color. No functional changes."),
]

def expand_terms(concept: str) -> list[str]:
    terms = [concept]
    for tech in kg.predecessors(concept):
        terms.append(tech)
    return terms

expanded = expand_terms("结账服务")
assert "Cart-Flow-V2" in expanded

relevant = [d.page_content for d in docs if any(t in d.page_content for t in expanded)]
assert any("Error Rate" in s for s in relevant)
```

---

## 09 矛盾事实（时间冲突 / Latest Truth）

### 痛点需求

知识库存在新旧版本并存（政策、制度、配置），系统必须以确定性方式输出“当前有效版本”，并能解释裁决依据。

### 处理方法

- 建模：将事实记录为带 `valid_from`（或 `effective_date`）的时间属性；同一主题维护历史序列。
- 查询：按时间排序，选择最新记录作为真值（可扩展为有效期区间与冲突策略）。

### 示例例子

远程办公政策：2021（5 天）→ 2023（2 天）→ 2024（0 天）。用户问“每周可远程办公几天？”

### 示例代码（保留原示例的核心逻辑，并补齐可运行数据）

```python
import dateparser
import networkx as nx

docs = [
    Document(page_content="[2021] 根据 Flex-21 计划，所有员工每周可享 5 天远程办公。"),
    Document(page_content="[2023] 因返岗办公（RTO）要求，远程办公额度缩减至 2 天。"),
    Document(page_content="[2024] 即日起，全面取消全职远程办公。最大额度为 0 天。"),
]

kg = nx.DiGraph()

def add_temporal_fact(topic: str, value: str, date_str: str, source: str) -> None:
    dt = dateparser.parse(date_str)
    if not kg.has_node(topic):
        kg.add_node(topic, history=[])
    kg.nodes[topic]["history"].append({"value": value, "date": dt, "source": source})

for d in docs:
    # 对齐原文的抽取目标：主题 | 取值 | 日期（此处用规则模拟抽取结果）
    if "远程办公" in d.page_content:
        year = d.page_content.split("]")[0].strip("[")
        if "5 天" in d.page_content:
            add_temporal_fact("远程办公政策", "5 天/周", year, d.page_content)
        elif "2 天" in d.page_content:
            add_temporal_fact("远程办公政策", "2 天/周", year, d.page_content)
        elif "0 天" in d.page_content:
            add_temporal_fact("远程办公政策", "0 天/周", year, d.page_content)

def resolve_latest_truth(topic: str) -> dict:
    history = kg.nodes[topic]["history"]
    return sorted(history, key=lambda x: x["date"], reverse=True)[0]

latest = resolve_latest_truth("远程办公政策")
assert latest["value"] == "0 天/周"
```

---

## 10 因果综合（规则检查 / Logic & Computation）

### 痛点需求

风险由“多个条件同时成立”触发（化学反应、合规规则、依赖冲突）时，系统必须执行确定性规则推断，而非依赖 LLM 自由生成。

### 处理方法

- 建模：把“成分/条件”结构化为节点与关系。
- 推理：实现规则引擎（最小可用：一组可测试的 if/then 规则；生产可替换为 Datalog/规则库）。

### 示例例子

存放 `84 消毒液（次氯酸钠）` 与 `洁厕灵（盐酸）` → 混合产生有毒气体 → 报警。

### 示例代码

```python
kg = nx.DiGraph()
kg.add_edge("84消毒液", "次氯酸钠", relation="HAS_INGREDIENT")
kg.add_edge("洁厕灵", "盐酸", relation="HAS_INGREDIENT")

def is_dangerous_mix(a: str, b: str) -> bool:
    ingredients_a = set(kg.successors(a))
    ingredients_b = set(kg.successors(b))
    return ("次氯酸钠" in ingredients_a) and ("盐酸" in ingredients_b)

assert is_dangerous_mix("84消毒液", "洁厕灵") is True
```

---

## 11 否定与缺失（Set Difference）

### 痛点需求

用户问“哪些产品不含花生？”时，正确答案往往来自“不提花生”的安全产品。系统必须支持“缺失即条件”的确定性判定与可解释输出。

### 处理方法

- 建图：产品与成分/污染风险的关系（`CONTAINS`、`TRACES_OF`）。
- 查询：安全集合 = 全部产品集合 − 不安全集合（含花生或存在花生污染风险）。

### 示例例子

能量棒中：坚果脆脆含花生；莓果爆有花生同线风险；可可喜悦未提花生且无风险 → 安全。

### 示例代码（保留原示例的核心逻辑）

```python
kg = nx.DiGraph()
kg.add_node("坚果脆脆", type="Product")
kg.add_node("可可喜悦", type="Product")
kg.add_node("莓果爆", type="Product")

kg.add_edge("坚果脆脆", "花生", relation="CONTAINS")
kg.add_edge("莓果爆", "花生", relation="TRACES_OF")

def safe_products(allergen: str) -> set[str]:
    all_products = {n for n, d in kg.nodes(data=True) if d.get("type") == "Product"}
    unsafe = set()
    for node in [n for n in kg.nodes() if allergen in n]:
        unsafe.update(set(kg.predecessors(node)))
    return all_products - unsafe

assert safe_products("花生") == {"可可喜悦"}
```

---

## 12 聚合（计数/统计 / Deterministic Aggregation）

### 痛点需求

当用户问“有多少唯一供应商/有多少一级供应商”时，系统必须返回可审计、可复现的确定性计数，并处理别名/重复提及。

### 处理方法

- 实体归一化（Entity Normalization）：`Apex Corp` 与 `Apex Inc` → 同一规范实体 `apex`。
- 聚合：对图节点集合做 `COUNT(DISTINCT ...)`（示例用 `len(set(...))`）。

### 示例例子

`Apex Corp`、`Apex Inc`、`Beta‑Tech`、`Gamma Logistics` 参与宙斯计划；需要输出唯一 Tier‑1 供应商数量。

### 示例代码（保留原示例的核心逻辑，并补齐数据）

```python
kg = nx.DiGraph()

def normalize_entity(name: str) -> str:
    s = name.lower().strip()
    for suf in (" corp", " inc", " ltd", " llc"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip()

project = normalize_entity("Project Zeus")
suppliers = ["Apex Corp", "Apex Inc", "Beta-Tech", "Gamma Logistics"]
for supplier in suppliers:
    kg.add_edge(project, normalize_entity(supplier), relation="HAS_SUPPLIER")

unique_suppliers = set(kg.successors(project))
assert len(unique_suppliers) == 3  # apex / beta-tech / gamma logistics
```

## 5. 生产落地建议（将示例升级为可维护系统）

1. **Schema 先行**：定义实体类型与允许关系集（`ALLOWED_RELATIONS`）并纳入版本控制；抽取与入库必须可验证、可回放。
2. **图谱作为“事实与计算层”**：对“是否存在关系”“方向”“计数”“集合差”“时间裁决”等，优先走确定性图查询/计算。
3. **LLM 仅做抽取与表述**：抽取输出必须结构化；最终表述可由 LLM 生成，但结论应来自图查询/规则引擎输出。
4. **可观测与审计**：每个结论应能追溯到图上的节点/边与来源文档（source/valid_from 等元数据）。


## 参考Schema实现

https://github.com/HKUST-KnowComp/AutoSchemaKG

https://arxiv.org/abs/2408.05357