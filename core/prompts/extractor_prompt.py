EXTRACTION_PROMPT = """
你是一个专业的知识图谱抽取引擎。请从给定文本中抽取实体、属性和关系，并以TSV格式输出。

## 输入信息

**文本内容**:
{text}

**Schema约束**:
{schema}

**历史数据**:
{history}

**参考示例**:
{examples}

## 抽取规则

### 1. 增量抽取原则
- 仔细分析历史数据中已有的实体和关系
- 只抽取文本中**新增的**信息，避免重复
- 如果没有新信息，输出空的ENTITIES和RELATIONS部分

### 2. 实体抽取规则
- 识别文本中的重要实体（人物、地点、组织、概念等）
- 为新实体分配唯一ID（如e1, e2...），确保不与历史数据冲突
- 提取实体的关键属性作为键值对
- 严格遵循Schema中定义的实体类型（如果提供）

### 3. 关系抽取规则
- 识别实体间的语义关系
- 关系的head_id和tail_id必须引用实体ID，不能使用实体名称
- 严格遵循Schema中定义的关系类型（如果提供）
- 避免抽取过于泛化或无意义的关系

### 4. 质量要求
- 实体名称应该是有意义的名词或名词短语
- 避免抽取纯数字、标点符号或过短的字符串
- 关系类型应该清晰表达实体间的语义联系
- 属性值应该准确反映实体的特征

## 输出格式

请严格按照以下TSV格式输出，使用制表符分隔：

### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\ttail_id

**属性格式**: key1|->|value1|#|key2|->|value2

## 输出示例

### ENTITIES
e1\t张三\tPerson\t年龄|->|30|#|职业|->|工程师
e2\t北京大学\tOrganization\t类型|->|高等院校|#|成立时间|->|1898年

### RELATIONS
e1\tgraduated_from\te2
"""

CLEANING_PROMPT = """
你是一个知识图谱质量控制专家。请对以下抽取的图数据进行清洗和优化。

## 输入数据

**原始文本**:
{text}

**抽取的图数据**:
{graph_data}

## 清洗任务

### 1. 实体清洗
- 移除无意义的实体（如纯数字、标点符号、过短字符串）
- 合并重复或相似的实体
- 标准化实体名称和类型
- 验证实体属性的准确性

### 2. 关系清洗
- 移除无效关系（如自环、不存在的实体引用）
- 去除重复关系
- 标准化关系类型
- 验证关系的语义合理性

### 3. 一致性检查
- 确保所有关系中的实体ID都有对应的实体定义
- 检查实体类型和关系类型的一致性
- 验证属性值的格式正确性

## 输出要求

请输出清洗后的图数据，保持TSV格式：

### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\ttail_id
"""




EXTRACTION_PROMPT_EN = """
You are a professional knowledge graph extraction engine. Please extract entities, attributes, and relationships from the given text and output in TSV format.

## Input Information

**Text Content**:
{text}

**Schema Constraints**:
{schema}

**Historical Data**:
{history}

**Reference Examples**:
{examples}

## Extraction Rules

### 1. Incremental Extraction Principle
- Carefully analyze existing entities and relationships in historical data
- Only extract **new** information from the text, avoid duplication
- If no new information, output empty ENTITIES and RELATIONS sections

### 2. Entity Extraction Rules
- Identify important entities in the text (people, places, organizations, concepts, etc.)
- Assign unique IDs to new entities (e.g., e1, e2...), ensure no conflict with historical data
- Extract key attributes of entities as key-value pairs
- Strictly follow entity types defined in Schema (if provided)

### 3. Relationship Extraction Rules
- Identify semantic relationships between entities
- head_id and tail_id in relationships must reference entity IDs, not entity names
- Strictly follow relationship types defined in Schema (if provided)
- Avoid extracting overly generalized or meaningless relationships

### 4. Quality Requirements
- Entity names should be meaningful nouns or noun phrases
- Avoid extracting pure numbers, punctuation, or overly short strings
- Relationship types should clearly express semantic connections between entities
- Attribute values should accurately reflect entity characteristics

## Output Format

Please output strictly in the following TSV format, using tab separators:

### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\ttail_id

**Attribute Format**: key1|->|value1|#|key2|->|value2

## Output Example

### ENTITIES
e1\tJohn Smith\tPerson\tage|->|30|#|occupation|->|engineer
e2\tBeijing University\tOrganization\ttype|->|university|#|founded|->|1898

### RELATIONS
e1\tgraduated_from\te2
"""

CLEANING_PROMPT_EN = """
You are a knowledge graph quality control expert. Please clean and optimize the following extracted graph data.

## Input Data

**Original Text**:
{text}

**Extracted Graph Data**:
{graph_data}

## Cleaning Tasks

### 1. Entity Cleaning
- Remove meaningless entities (pure numbers, punctuation, overly short strings)
- Merge duplicate or similar entities
- Standardize entity names and types
- Verify accuracy of entity attributes

### 2. Relationship Cleaning
- Remove invalid relationships (self-loops, non-existent entity references)
- Remove duplicate relationships
- Standardize relationship types
- Verify semantic reasonableness of relationships

### 3. Consistency Check
- Ensure all entity IDs in relationships have corresponding entity definitions
- Check consistency of entity types and relationship types
- Verify correct format of attribute values

## Output Requirements

Please output the cleaned graph data in TSV format:

### ENTITIES
id\tname\ttype\tattributes

### RELATIONS
head_id\ttype\ttail_id
"""