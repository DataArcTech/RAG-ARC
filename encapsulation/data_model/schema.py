from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class GraphData:
    """Unified graph data structure"""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[List[str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity_id: str, name: str, entity_type: str, attributes: Dict[str, Any] = None):
        """Add entity"""
        entity = {
            'id': entity_id,
            'entity_name': name,
            'entity_type': entity_type,
            'attributes': attributes or {}
        }
        self.entities.append(entity)

    def add_relation(self, head_id: str, relation_type: str, tail_id: str):
        """Add relation"""
        self.relations.append([head_id, relation_type, tail_id])

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID"""
        for entity in self.entities:
            if entity.get('id') == entity_id:
                return entity
        return None

    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get entity by name"""
        for entity in self.entities:
            if entity.get('entity_name') == name:
                return entity
        return None

    def is_empty(self) -> bool:
        """Check if graph is empty"""
        return len(self.entities) == 0 and len(self.relations) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'entities': self.entities,
            'relations': self.relations,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphData':
        """Create GraphData from dictionary"""
        return cls(
            entities=data.get('entities', []),
            relations=data.get('relations', []),
            metadata=data.get('metadata', {})
        )

@dataclass
class Chunk:
    """Chunk data structure"""
    content: str
    owner_id: Optional[str] = None  # User ID (UUID string format) for access control
    domain: Optional[str] = None  # Domain/category for index partitioning and filtering
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    graph: Optional[GraphData] = None

    def __post_init__(self):
        if self.graph is None:
            self.graph = GraphData()
