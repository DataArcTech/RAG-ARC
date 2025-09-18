from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class GraphData:
    """统一的图数据结构"""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity_id: str, name: str, entity_type: str, attributes: Dict[str, Any] = None):
        """添加实体"""
        entity = {
            'id': entity_id,
            'entity_name': name,
            'entity_type': entity_type,
            'attributes': attributes or {}
        }
        self.entities.append(entity)

    def add_relation(self, head_id: str, relation_type: str, tail_id: str):
        """添加关系"""
        self.relations.append([head_id, relation_type, tail_id])

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取实体"""
        for entity in self.entities:
            if entity.get('id') == entity_id:
                return entity
        return None

    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取实体"""
        for entity in self.entities:
            if entity.get('entity_name') == name:
                return entity
        return None

    def is_empty(self) -> bool:
        """检查是否为空图"""
        return len(self.entities) == 0 and len(self.relations) == 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'entities': self.entities,
            'relations': self.relations,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphData':
        """从字典创建GraphData"""
        return cls(
            entities=data.get('entities', []),
            relations=data.get('relations', []),
            metadata=data.get('metadata', {})
        )

@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    graph: Optional[GraphData] = None

    def __post_init__(self):
        if self.graph is None:
            self.graph = GraphData()
