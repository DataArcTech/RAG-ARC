from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class Event(BaseModel):
    """事件模型"""
    id: str = Field(..., description="事件唯一ID，例如 event_0", pattern=r"^event_\d+$")
    content: str = Field(..., description="事件内容")
    type: str = Field(..., description="事件类型")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建Event对象"""
        return cls(**data)


class Entity(BaseModel):
    """实体模型"""
    id: str = Field(..., description="实体唯一ID，例如 entity_0", pattern=r"^entity_\d+$")
    entity_name: str = Field(..., description="实体文本")
    entity_type: Literal["资源", "属性", "方法", "环境"] = Field(..., description="实体类别")
    entity_description: str = Field(..., description="实体内容说明")
    event_indices: List[int] = Field(default_factory=list, description="实体关联的事件索引")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """从字典创建Entity对象"""
        return cls(**data)


class EventRelation(BaseModel):
    """事件关系模型""" 
    id: str = Field(..., description="事件关系唯一ID，例如 event_relation_0", pattern=r"^event_relation_\d+$")
    head_id: str = Field(..., description="关系头事件ID", pattern=r"^event_\d+$")
    tail_id: str = Field(..., description="关系尾事件ID", pattern=r"^event_\d+$")
    relation_type: Literal["时序关系", "因果关系", "层级关系", "条件关系"]
    description: Optional[str] = Field(None, description="关系证据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventRelation':
        """从字典创建EventRelation对象"""
        return cls(**data)


class EntityRelation(BaseModel):
    """实体关系模型"""
    id: str = Field(..., description="实体关系唯一ID，例如 entity_relation_0", pattern=r"^entity_relation_\d+$")
    head_id: str = Field(..., description="头实体ID", pattern=r"^entity_\d+$")
    tail_id: str = Field(..., description="尾实体ID", pattern=r"^entity_\d+$")
    relation_type: Literal["包含关系", "属性关系", "定位关系", "实例关系", "遵循关系", "时间关系"]
    description: str = Field(..., description="关系证据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRelation':
        """从字典创建EntityRelation对象"""
        return cls(**data)


class KnowledgeStructure(BaseModel):
    events: List[Event] = []
    entities: List[Entity] = []
    entity_relations: List[EntityRelation] = []
    event_relations: List[EventRelation] = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "events": [event.to_dict() for event in self.events],
            "entities": [entity.to_dict() for entity in self.entities],
            "event_relations": [relation.to_dict() for relation in self.event_relations],
            "entity_relations": [relation.to_dict() for relation in self.entity_relations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeStructure':
        """从字典创建KnowledgeStructure对象"""
        return cls(
            events=[Event.from_dict(event) for event in data.get("events", [])],
            entities=[Entity.from_dict(entity) for entity in data.get("entities", [])],
            event_relations=[EventRelation.from_dict(relation) for relation in data.get("event_relations", [])],
            entity_relations=[EntityRelation.from_dict(relation) for relation in data.get("entity_relations", [])]
        )


class EntityList(BaseModel):
    """用于LLM响应格式化的提及列表类"""
    entities: List[Entity]

    def __len__(self):
        return len(self.entities)



TCL_PROMPT = """
**角色与目标 (Role & Goal)**
你是一个专注于 **工业领域** 的图结构信息提取系统，尤其擅长处理 **生产流程、质量控制、产品手册、安全规程** 等文档。你的核心任务是：基于当前输入文本和历史提取记录，执行一次**增量信息提取**，仅输出本次新发现的知识元素。

**输入格式 (Input Specification)**
你将接收到以下两部分输入：

1.  `current_text`: `{text}` - 本轮需要分析处理的文本段落。
2.  `extraction_history`: `{history}` - 一个 `KnowledgeStructure` 格式的JSON对象，包含了所有先前轮次已提取的信息。首次执行时，此对象可能为空。

**核心指令：增量提取原则 (Primary Directive: Incremental Extraction)**
你的所有操作都必须遵循增量原则。最终输出**只应包含**相对于 `extraction_history` 的**新增内容**。
**ID唯一性校验**: 在将任何元素（事件、实体、关系）加入最终输出之前，必须严格对照 `extraction_history` 进行检查，确保其ID唯一性。

**唯一性校验 (Uniqueness Check)**:
    * 在将任何元素（事件、实体、关系）加入最终输出之前，必须严格对照 `extraction_history` 进行检查，确保其唯一性。
    * **跳过（不输出）** 任何已存在于历史中的信息。判定标准如下：
        * **实体 (Entity)**: `entity_name` 和 `entity_type` 的组合在历史中已存在。
        * **事件 (Event)**: `content`, `type` 的组合在历史中已存在。
        * **实体关系 (Entity Relation)**: `head_id`, `tail_id`, 和 `relation_type` 的三元组在历史中已存在。
        * **事件关系 (Event Relation)**: `head_id`, `tail_id`, 和 `relation_type` 的三元组在历史中已存在。

**内部推理流程：思维链工作流程(Internal Reasoning: Chain-of-Thought Workflow)**
严格遵循以下五个步骤进行分析和推理。

**步骤一：识别出所有“事件” (Events)**
  * **任务**: 找出 `current_text` 中所有的**关键活动、指令、规范、流程步骤等**。
  * **标准**: 事件必须体现“做了什么/要求做什么/发生了什么”。
  * **类型**: 根据事件的核心动词或动作，为其生成一个简洁、准确的名词化标签作为其 `type`。

**步骤二：识别并规范化所有“实体” (Entities)**
  * **任务**: 找出 `current_text` 中所有属于核心概念的名词或名词短语，并关联到事件。
  * **类型**: 仅限 `资源 (Resource)`, `属性 (Property)`, `方法 (Method)`, `环境 (Environment)`。
  * **操作**: 提取每个实体都的确定一个统一的规范化名称（entity_name），并标注其类型（entity_type）。

**步骤三：建立“实体”之间的显式关系 (Entity Relations)**
  * **任务**: 回顾上一步识别出的实体列表，专注分析实体之间的静态、固有联系。
  * **标准**: 关系必须由明确的语言结构（如所有格“的”、介词短语等）或上下文逻辑直接支撑。
  * **限定关系类型**: 仅从以下类型中选择：
      * `包含关系 (contains)`: 一个实体在物理上或概念上包含另一个实体。
      * `属性关系 (has_property)`: 一个实体拥有另一个实体作为其属性或特征。
      * `定位关系 (located_at)`: 一个实体位于另一个实体所代表的环境中。
      * `实例关系 (is_instance_of)`: 一个具体实体是某个抽象概念的实例。
      * `遵循关系 (follows_standard)`: 某个资源或方法遵循某个标准或规范。
      * `时间关系 (time_relation)`: 两个实体之间存在时间上的先后顺序或同时性。
  * **示例导引**: 文本"`压缩机的额定功率是1.5kW`"应识别出实体 `压缩机`，ID `entity_0`，`额定功率`，ID `entity_1` 和 `{{"head_id": "entity_0", "tail_id": "entity_1", "relation_type": "属性关系"}}` 的关系。

**步骤四：建立“事件”之间的逻辑关系 (Event Relations)**
  * **任务**: 回顾步骤一的事件列表，寻找连接它们的**明确**逻辑关系。
  * **类型**: 仅限 `层级关系 (hierarchical)`, `时序关系 (sequential)`, `因果关系 (causal)`, `条件关系 (conditional)`。

**输出格式 (Output Specification)**
在完成上述所有内部推理后，将所有**相对历史的新增内容**组装成一个结构化的KnowledgeStructure Python Pydantic模型。

    * **你的最终输出必须且只能是一个符合KnowledgeStructure结构的Python Pydantic模型。**
    * **如果 `current_text` 未包含任何新增信息，必须返回一个所有列表均为空的空结构。**
"""


# 通用领域的清洗提示词（可扩展）
TCL_CLEAN_PROMPT = """
## 📥 输入数据

你将收到以下两项输入：
* `document_content`: 原始文档内容
{document_content}
* `entities_json`: 原始实体提及列表
{entities_json}
---
## 🧹 清洗与规范化规则
###1. 删除无用实体（整条 entity 被移除）
* `entity_name` 为纯数字、日期时间、标点、代词、连接词、无义短语、纯动词、纯形容词
* `entity_name` 含无效信息，如：“版本C”、“型号F”、“编号I”等缺乏识别性的名称
* `entity_description` 无内容或为空泛（如“设备的一种”、“某种装置”）
* `entity_type` 不属于允许的类型（仅保留 `资源`、`属性`、`方法`、`环境`）
---
###2. 实体名称规范化
对每条 entity 的 `entity_name` 执行以下操作：
* 去除冗余修饰词（如“产品”、“装置”、“一种”、“类型”、“某某”等）
* 合并指向同一概念的entity，将指向同一概念的entity的`entity_name`合并为同一个,并保留所有原始的`id`和`entity_type`

---
###3. 实体描述清洗（`entity_description`）

* 删除冗余前缀或泛化表达
* 可参考 `document_content` 进行补充，但必须准确简洁
* 若无法提供有价值描述，允许将其设为空字符串或删除该字段

---
## 输出格式要求

* 输出为一个 JSON 数组（列表），每项为一个实体 entity
* 每个 entity 必须保留字段：`id`, `entity_name`, `entity_type`, `entity_description`，`event_indices`
* 保留所有清洗合格的 entity
* 删除无效或无意义的 entity 项
* 对于同义实体，仅需统一 `entity_name`，但原始的 `id` 必须各自保留
"""