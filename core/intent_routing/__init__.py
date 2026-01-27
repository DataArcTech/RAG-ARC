"""Core intent routing primitives.

This package is framework-agnostic. Application-layer wiring (Redis/session handling,
embedding provider construction, and RAG pipeline decisions) lives under
`application/intent_routing/`.
"""

from core.intent_routing.semantic_router import (
    EmbeddingProvider,
    IntentDecision,
    IntentMatch,
    IntentRoute,
    SemanticIntentRouter,
)
from core.intent_routing.topic_stack import TopicSelection, TopicStack, TopicStackSettings

__all__ = [
    "EmbeddingProvider",
    "IntentDecision",
    "IntentMatch",
    "IntentRoute",
    "SemanticIntentRouter",
    "TopicSelection",
    "TopicStack",
    "TopicStackSettings",
]

