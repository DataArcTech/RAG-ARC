from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None



