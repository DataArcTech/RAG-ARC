# EXTRACTION_PROMPT = """
# 你是一个专业的知识图谱抽取引擎。请从给定文本中抽取实体、属性和关系，并以TSV格式输出。

# ## 输入信息

# **文本内容**:
# {text}

# **Schema约束**:
# {schema}

# **历史数据**:
# {history}

# **参考示例**:
# {examples}

# ## 抽取规则

# ### 1. 增量抽取原则
# - 仔细分析历史数据中已有的实体和关系
# - 只抽取文本中**新增的**信息，避免重复
# - 如果没有新信息，输出空的ENTITIES和RELATIONS部分

# ### 2. 实体抽取规则
# - 识别文本中的重要实体（人物、地点、组织、概念等）
# - 为新实体分配唯一ID（如e1, e2...），确保不与历史数据冲突
# - 提取实体的关键属性作为键值对
# - 严格遵循Schema中定义的实体类型（如果提供）

# ### 3. 关系抽取规则
# - 识别实体间的语义关系
# - 关系的head_id和tail_id必须引用实体ID，不能使用实体名称
# - 严格遵循Schema中定义的关系类型（如果提供）
# - 避免抽取过于泛化或无意义的关系

# ### 4. 质量要求
# - 实体名称应该是有意义的名词或名词短语
# - 避免抽取纯数字、标点符号或过短的字符串
# - 关系类型应该清晰表达实体间的语义联系
# - 属性值应该准确反映实体的特征

# ## 输出格式

# 请严格按照以下TSV格式输出，使用制表符分隔：

# ### ENTITIES
# id\tname\ttype\tattributes

# ### RELATIONS
# head_id\ttype\ttail_id

# **属性格式**: key1|->|value1|#|key2|->|value2

# ## 输出示例

# ### ENTITIES
# e1\t张三\tPerson\t年龄|->|30|#|职业|->|工程师
# e2\t北京大学\tOrganization\t类型|->|高等院校|#|成立时间|->|1898年

# ### RELATIONS
# e1\tgraduated_from\te2
# """


EXTRACTION_PROMPT = """
你是一个知识图谱抽取引擎，负责将文本转化为结构化的TSV数据。
## 抽取原则与目标
本次任务的目标是将文本转化为结构化知识图谱。抽取应遵循以下原则：
1.  **结构优先**: 优先抽取能构成文本核心事实框架的关系(Relations)，作为图谱的结构基础。
2.  **属性补充**: 围绕已识别的实体，抽取其描述性属性(Attributes)，以完善信息细节。
3.  **信息保真**: 最终的图谱应准确反映原文的核心信息。
---
## 输入信息
**文本内容**:
{text}
**Schema约束**:
{schema}
**历史数据**:
{history}
**参考示例**:
{examples}
---
## 抽取指令
1.  **增量抽取**: 仔细比对**历史数据**，仅当**当前文本**中出现了建立新关系、或补充新实体/属性的信息时，才进行抽取。
2.  **实体与属性抽取**:
    * **实体规则**:
        * 识别关系框架中的核心参与者作为实体。
        * 分配不与历史数据冲突的唯一ID (e.g., e1, e2...)。
        * 实体名称为名词或名词短语。
        * 实体类型遵循Schema。
        * **实体类型定义**: `entity_type` 必须是通用、抽象的分类。任何具体的角色、职业或业务描述都应作为 `attributes` 抽取，而非类型。
    * **属性规则**:
        * 抽取实体必要的描述性特征。
        * 属性值应直接采用原文中的表述。
        * 避免将关系抽取为属性。
        * 格式: `key|->|value`，多个属性用`|#|`分隔。
3.  **关系抽取**:
    * 优先抽取体现文本主旨的核心关系。
    * 关系类型遵循Schema。若无，尽可能从原文中抽取。
    * `head_id`和`tail_id`必须使用实体ID。

## 输出格式 (TSV)
### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\ttail_id
---
## 输出示例
### ENTITIES
e1\t张三\t人物\t年龄|->|30|#|职业|->|工程师
e2\t北京大学\t组织\t类型|->|高等院校|#|成立时间|->|1898年

### RELATIONS
e1\t毕业于\te2
"""


CLEANING_PROMPT = """
你是一个知识图谱质量控制与优化引擎。

## 核心目标
以输入的**原始文本**为唯一事实来源，对**抽取的图数据**进行审查和修正，旨在提升图谱的**准确性、一致性和规范性**。
---
## 输入数据
**原始文本**:
{text}

**抽取的图数据**:
{graph_data}
---
## 清洗与优化指令
1.  **真实性校验 (Fact Verification)**:
    * 以`原始文本`为基准，核实并修正所有实体、属性及关系，确保信息准确。删除文本中不存在的虚构信息。

2.  **实体规范化 (Entity Normalization)**:
    * **合并**: 将指代同一对象的重复或相似实体进行合并（选择最准确或最完整的名称）。
    * **清洗**: 移除无意义的实体（如：标点符号、无关的纯数字、过短的碎屑字符）。
    * **标准化**: 统一实体命名（如：使用官方全称代替简称）和实体类型。

3.  **关系完整性 (Relationship Integrity)**:
    * **去重**: 移除语义和指向完全重复的关系。
    * **修正**: 删除无效连接，例如：
        * 指向不存在实体ID的关系（悬空关系）。
        * 实体指向自己的关系（自环）。
    * **校验**: 根据常识和文本内容，验证关系的语义合理性。

4.  **一致性与格式 (Consistency & Formatting)**:
    * 确保实体与关系类型符合逻辑（例如，一个`Person`类型的实体不应有`founded_in`关系）。
    * 检查并修正`attributes`字段的格式，确保其符合`key|->|value|#|...`的规范。
---
## 输出要求
请输出清洗优化后的图数据，并严格保持原始的TSV格式。

### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\tail_id
"""



# EXTRACTION_PROMPT_EN = """
# You are a professional knowledge graph extraction engine. Please extract entities, attributes, and relationships from the given text and output in TSV format.

# ## Input Information

# **Text Content**:
# {text}

# **Schema Constraints**:
# {schema}

# **Historical Data**:
# {history}

# **Reference Examples**:
# {examples}

# ## Extraction Rules

# ### 1. Incremental Extraction Principle
# - Carefully analyze existing entities and relationships in historical data
# - Only extract **new** information from the text, avoid duplication
# - If no new information, output empty ENTITIES and RELATIONS sections

# ### 2. Entity Extraction Rules
# - Identify important entities in the text (people, places, organizations, concepts, etc.)
# - Assign unique IDs to new entities (e.g., e1, e2...), ensure no conflict with historical data
# - Extract key attributes of entities as key-value pairs
# - Strictly follow entity types defined in Schema (if provided)

# ### 3. Relationship Extraction Rules
# - Identify semantic relationships between entities
# - head_id and tail_id in relationships must reference entity IDs, not entity names
# - Strictly follow relationship types defined in Schema (if provided)
# - Avoid extracting overly generalized or meaningless relationships

# ### 4. Quality Requirements
# - Entity names should be meaningful nouns or noun phrases
# - Avoid extracting pure numbers, punctuation, or overly short strings
# - Relationship types should clearly express semantic connections between entities
# - Attribute values should accurately reflect entity characteristics

# ## Output Format

# Please output strictly in the following TSV format, using tab separators:

# ### ENTITIES
# id\tname\ttype\tattributes

# ### RELATIONS
# head_id\ttype\ttail_id

# **Attribute Format**: key1|->|value1|#|key2|->|value2

# ## Output Example

# ### ENTITIES
# e1\tJohn Smith\tPerson\tage|->|30|#|occupation|->|engineer
# e2\tBeijing University\tOrganization\ttype|->|university|#|founded|->|1898

# ### RELATIONS
# e1\tgraduated_from\te2
# """

CLEANING_PROMPT_EN = """
You are a knowledge graph quality control and optimization engine.
## Core Objective
Using the input **Original Text** as the single source of truth, review and correct the **Extracted Graph Data**. The objective is to enhance the graph's **accuracy, consistency, and standardization**.
---
## Input Data
**Original Text**:
{text}

**Extracted Graph Data**:
{graph_data}
---
## Cleaning and Optimization Instructions
1.  **Fact Verification**:
    * Using the `Original Text` as the baseline, verify and correct all entities, attributes, and relationships to ensure accuracy. Remove any fabricated information not present in the text.

2.  **Entity Normalization**:
    * **Merge**: Merge duplicate or similar entities that refer to the same object (choose the most accurate or complete name).
    * **Clean**: Remove meaningless entities (e.g., punctuation, irrelevant numbers, short character fragments).
    * **Standardize**: Unify entity naming (e.g., use official full names instead of acronyms) and entity types.
    

3.  **Relationship Integrity**:
    * **Deduplicate**: Remove relationships that are semantically and directionally identical.
    * **Correct**: Delete invalid links, such as:
        * Relationships pointing to non-existent entity IDs (dangling edges).
        * Relationships where an entity points to itself (self-loops).
    * **Verify**: Check the semantic reasonableness of relationships based on common sense and the text.

4.  **Consistency & Formatting**:
    * Ensure entity and relationship types are logical (e.g., an entity of type 'Person' should not have a 'founded_in' relationship).
    * Check and correct the format of the `attributes` field to ensure it conforms to the `key|->|value|#|...` specification.
---
## Output Requirements
Please output the cleaned and optimized graph data, strictly maintaining the original TSV format.

### ENTITIES
id\tname\type\tattributes

### RELATIONS
head_id\ttype\ttail_id
"""



EXTRACTION_PROMPT_EN = """
You are a knowledge graph extraction engine responsible for converting text into structured TSV data.

## Extraction Principles and Goal
The goal of this task is to convert text into a structured knowledge graph. The extraction should adhere to the following principles:
1.  **Structure-First**: Prioritize extracting relations that form the core factual framework of the text, serving as the graph's structural foundation.
2.  **Attribute Supplementation**: Around the identified entities, extract their descriptive attributes to complete the informational details.
3.  **Information Fidelity**: The final graph must accurately reflect the core information of the original text.
---
## Input Information
**Text Content**:
{text}

**Schema Constraints**:
{schema}

**Historical Data**:
{history}

**Reference Examples**:
{examples}
---
## Extraction Instructions
1.  **Incremental Extraction**: Carefully compare against **Historical Data**. Only extract information when the **current text** introduces new relationships or supplements new entities/attributes.

2.  **Entity and Attribute Extraction**:
    * **Entity Rules**:
        * Identify core participants within the relationship framework as entities.
        * Assign unique IDs (e.g., e1, e2...) that do not conflict with historical data.
        * Entity names must be nouns or noun phrases.
        * Entity types must adhere to the Schema.
        * Entity types: `entity_type` must be generic, high-level categories. Any specific roles, occupations, or business descriptions should be extracted as attributes rather than types.
    * **Attribute Rules**:
        * Extract necessary descriptive features for each entity.
        * Attribute values should be directly taken from the original text.
        * Avoid extracting relationships as attributes.
        * Format: Strictly use `key|->|value`, with `|#|` separating multiple attributes.

3.  **Relationship Extraction**:
    * Prioritize extracting core relationships that reflect the main subject of the text.
    * Relationship types must adhere to the Schema. If not, extract from the original text.
    * `head_id` and `tail_id` must use entity IDs.

## Output Format (TSV)
### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\ttail_id
---
## Output Example
### ENTITIES
e1\tJohn Smith\tPerson\tage|->|30|#|occupation|->|engineer
e2\tBeijing University\tOrganization\ttype|->|university|#|founded|->|1898

### RELATIONS
e1\tgraduated from\te2
"""


# Simplified prompts for single-round extraction (max_rounds == 1)
# These prompts omit history and incremental extraction logic to reduce token usage

EXTRACTION_PROMPT_SIMPLE = """
你是一个知识图谱抽取引擎，负责将文本转化为结构化的TSV数据。

## 抽取原则与目标
本次任务的目标是将文本转化为结构化知识图谱。抽取应遵循以下原则：
1.  **结构优先**: 优先抽取能构成文本核心事实框架的关系(Relations)，作为图谱的结构基础。
2.  **属性补充**: 围绕已识别的实体，抽取其描述性属性(Attributes)，以完善信息细节。
3.  **信息保真**: 最终的图谱应准确反映原文的核心信息。
---
## 输入信息
**文本内容**:
{text}

**Schema约束**:
{schema}

**参考示例**:
{examples}
---
## 抽取指令
1.  **实体与属性抽取**:
    * **实体规则**:
        * 识别关系框架中的核心参与者作为实体。
        * 分配唯一ID (e.g., e1, e2...)。
        * 实体名称为名词或名词短语。
        * 实体类型遵循Schema。
        * **实体类型定义**: `entity_type` 必须是通用、抽象的分类。任何具体的角色、职业或业务描述都应作为 `attributes` 抽取，而非类型。
    * **属性规则**:
        * 抽取实体必要的描述性特征。
        * 属性值应直接采用原文中的表述。
        * 避免将关系抽取为属性。
        * 格式: `key|->|value`，多个属性用`|#|`分隔。

2.  **关系抽取**:
    * 优先抽取体现文本主旨的核心关系。
    * 关系类型遵循Schema。若无，尽可能从原文中抽取。
    * `head_id`和`tail_id`必须使用实体ID。

## 输出格式 (TSV)
### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\ttail_id
---
## 输出示例
### ENTITIES
e1\t张三\t人物\t年龄|->|30|#|职业|->|工程师
e2\t北京大学\t组织\t类型|->|高等院校|#|成立时间|->|1898年

### RELATIONS
e1\t毕业于\te2
"""


EXTRACTION_PROMPT_SIMPLE_EN = """
You are a knowledge graph extraction engine responsible for converting text into structured TSV data.

## Extraction Principles and Goal
The goal of this task is to convert text into a structured knowledge graph. The extraction should adhere to the following principles:
1.  **Structure-First**: Prioritize extracting relations that form the core factual framework of the text, serving as the graph's structural foundation.
2.  **Attribute Supplementation**: Around the identified entities, extract their descriptive attributes to complete the informational details.
3.  **Information Fidelity**: The final graph must accurately reflect the core information of the original text.
---
## Input Information
**Text Content**:
{text}

**Schema Constraints**:
{schema}

**Reference Examples**:
{examples}
---
## Extraction Instructions
1.  **Entity and Attribute Extraction**:
    * **Entity Rules**:
        * Identify core participants within the relationship framework as entities.
        * Assign unique IDs (e.g., e1, e2...).
        * Entity names must be nouns or noun phrases.
        * Entity types must adhere to the Schema.
        * Entity types: `entity_type` must be generic, high-level categories. Any specific roles, occupations, or business descriptions should be extracted as attributes rather than types.
    * **Attribute Rules**:
        * Extract necessary descriptive features for each entity.
        * Attribute values should be directly taken from the original text.
        * Avoid extracting relationships as attributes.
        * Format: Strictly use `key|->|value`, with `|#|` separating multiple attributes.

2.  **Relationship Extraction**:
    * Prioritize extracting core relationships that reflect the main subject of the text.
    * Relationship types must adhere to the Schema. If not, extract from the original text.
    * `head_id` and `tail_id` must use entity IDs.

## Output Format (TSV)
### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\ttail_id
---
## Output Example
### ENTITIES
e1\tJohn Smith\tPerson\tage|->|30|#|occupation|->|engineer
e2\tBeijing University\tOrganization\ttype|->|university|#|founded|->|1898

### RELATIONS
e1\tgraduated from\te2
"""