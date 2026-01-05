from typing import Dict


DEFAULT_PPR_RELATION_WEIGHTS: Dict[str, float] = {
    "MENTIONS": 0.3,  # Lower weight for document mentions
    "RELATED_TO": 1.0,
    "PART_OF": 1.5,  # Higher weight for hierarchical relations
    "INSTANCE_OF": 1.3,
    "SIMILAR_TO": 0.8,
    "CONTAINS": 1.4,
    "DEPENDS_ON": 1.2,
    "SYNONYM": 0.6,  # Lower weight for synonym relations
}


__all__ = ["DEFAULT_PPR_RELATION_WEIGHTS"]

