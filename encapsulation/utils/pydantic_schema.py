from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Union


class Event(BaseModel):
    """事件模型"""
    id: str = Field(..., description="事件唯一ID，例如 event_0")
    content: str = Field(..., description="事件简要描述")
    type: str = Field(..., description="动作类型")
    participants: List[str] = Field(default_factory=list, description="参与的实体规范名")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建Event对象"""
        return cls(**data)


class Mention(BaseModel):
    """提及模型"""
    text: str = Field(..., description="原文字符串")
    entity_name: str = Field(..., description="规范化名称")
    # entity_type: Literal["Resource", "Attribute", "Method", "Environment"] = Field(..., description="Entity category")
    entity_type: Literal["题型", "考点", "解题方法", "考试模块"] = Field(..., description="实体类别")
    entity_description: Optional[str] = Field(None, description="简要描述")
    event_indices: List[int] = Field(default_factory=list, description="提及关联的事件索引")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mention':
        """从字典创建Mention对象"""
        return cls(**data)


class EventRelation(BaseModel):
    """事件关系模型"""
    head_event: str = Field(..., description="关系头事件ID")
    tail_event: str = Field(..., description="关系尾事件ID")
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
    head_entity: str = Field(..., description="实体A")
    tail_entity: str = Field(..., description="实体B")
    relation_type: str = Field(..., description="关系类型")
    description: Optional[str] = Field(None, description="关系证据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRelation':
        """从字典创建EntityRelation对象"""
        return cls(**data)


class KnowledgeStructure(BaseModel):
    """知识结构模型"""
    events: List[Event] = []
    event_relations: List[EventRelation] = []
    mentions: List[Mention] = []
    entity_relations: List[EntityRelation] = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "events": [event.to_dict() for event in self.events],
            "mentions": [mention.to_dict() for mention in self.mentions],
            "event_relations": [relation.to_dict() for relation in self.event_relations],
            "entity_relations": [relation.to_dict() for relation in self.entity_relations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeStructure':
        """从字典创建KnowledgeStructure对象"""
        return cls(
            events=[Event.from_dict(event) for event in data.get("events", [])],
            mentions=[Mention.from_dict(mention) for mention in data.get("mentions", [])],
            event_relations=[EventRelation.from_dict(relation) for relation in data.get("event_relations", [])],
            entity_relations=[EntityRelation.from_dict(relation) for relation in data.get("entity_relations", [])]
        )


class Entity(BaseModel):
    """实体模型"""
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型")
    description: Optional[str] = Field(None, description="实体描述")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """从字典创建Entity对象"""
        return cls(**data)


class Triple(BaseModel):
    """三元组模型"""
    head: str = Field(..., description="头实体")
    tail: str = Field(..., description="尾实体")
    relation: str = Field(..., description="关系类型")
    description: Optional[str] = Field(None, description="关系描述")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Triple':
        """从字典创建Triple对象"""
        return cls(**data)


class GraphTriples(BaseModel):
    """图三元组模型"""
    entities: List[Entity] = []
    relationships: List[Triple] = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [triple.to_dict() for triple in self.relationships]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphTriples':
        """从字典创建GraphTriples对象"""
        return cls(
            entities=[Entity.from_dict(entity) for entity in data.get("entities", [])],
            relationships=[Triple.from_dict(triple) for triple in data.get("relationships", [])]
        )


class MentionList(BaseModel):
    """用于LLM响应格式化的提及列表类"""
    mentions: List[Mention]

    def __len__(self):
        return len(self.mentions)


# Unified schema conversion helpers.
class PydanticUtils:
    """Pydantic helpers that provide consistent schema conversion methods."""
    
    @staticmethod
    def to_dict(obj: Union[BaseModel, Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Convert a Pydantic object (or list of Pydantic objects) into plain dicts.
        
        Args:
            obj: Pydantic object, dict, or a list containing these objects.
            
        Returns:
            A dict or list of dicts.
        """
        if isinstance(obj, list):
            return [PydanticUtils._convert_item(item) for item in obj]
        else:
            return PydanticUtils._convert_item(obj)
    
    @staticmethod
    def _convert_item(item: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert a single item."""
        if isinstance(item, BaseModel):
            return item.model_dump()
        elif isinstance(item, dict):
            return item
        else:
            return item
    
    @staticmethod
    def from_dict(cls: type, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[BaseModel, List[BaseModel]]:
        """
        Create Pydantic object(s) from dict payload(s).
        
        Args:
            cls: Pydantic model class.
            data: Dict payload or list of dict payloads.
            
        Returns:
            Pydantic object or list of objects.
        """
        if isinstance(data, list):
            return [cls(**item) for item in data]
        else:
            return cls(**data)
    
    @staticmethod
    def safe_get_attr(obj: Union[BaseModel, Dict[str, Any]], attr_name: str, default: Any = None) -> Any:
        """
        Safely get an attribute from a Pydantic object or dict.
        
        Args:
            obj: Pydantic object or dict.
            attr_name: Attribute key.
            default: Default value.
            
        Returns:
            The attribute value (or default).
        """
        if isinstance(obj, BaseModel):
            return getattr(obj, attr_name, default)
        elif isinstance(obj, dict):
            return obj.get(attr_name, default)
        return default
