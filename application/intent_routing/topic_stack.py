from dataclasses import dataclass
from typing import Any, Optional

from encapsulation.database.cache_db.redis_db import RedisDB
from core.intent_routing.topic_stack import TopicSelection, TopicStack, TopicStackSettings as TopicStackAlgoSettings


@dataclass(frozen=True)
class TopicStackSettings:
    """Application wiring settings for the session-scoped topic stack.

    Persistence is handled via Redis (derived state, TTL-based).
    The topic-stack algorithm itself lives in `core.intent_routing.topic_stack`.
    """

    enabled: bool
    redis_ttl_seconds: int
    max_user_messages: int
    max_topics: int
    same_topic_threshold: float
    return_to_topic_threshold: float
    history_mode: str
    sample_keep: int


class TopicStackManager:
    """Session-scoped topic stack stored in Redis.

    The stack is derived state and can be rebuilt from the latest N user messages.
    """

    _VERSION = TopicStack.VERSION

    def __init__(self, *, redis: RedisDB, settings: TopicStackSettings) -> None:
        self._redis = redis
        self._settings = settings
        self._algo = TopicStack(
            settings=TopicStackAlgoSettings(
                max_topics=int(settings.max_topics),
                same_topic_threshold=float(settings.same_topic_threshold),
                return_to_topic_threshold=float(settings.return_to_topic_threshold),
                sample_keep=int(settings.sample_keep),
            )
        )

    def _key(self, session_id: str) -> str:
        return f"intent_router:topic_stack:v{self._VERSION}:{session_id}"

    def load(self, session_id: str) -> Optional[dict[str, Any]]:
        if not self._settings.enabled:
            return None
        state = self._redis.get(self._key(session_id))
        return state if isinstance(state, dict) else None

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        if not self._settings.enabled:
            return
        payload = dict(state or {})
        payload.setdefault("v", self._VERSION)
        payload.setdefault("session_id", str(session_id))
        self._redis.set(self._key(session_id), payload, ttl=int(self._settings.redis_ttl_seconds))

    def rebuild_from_user_messages(
        self, *, session_id: str, user_messages: list[str], embeddings: list[list[float]]
    ) -> dict[str, Any]:
        """Rebuild topic stack state from the latest user messages.

        `user_messages` order must be chronological (oldest -> newest) for stability.
        """
        state = self._algo.rebuild_from_user_messages(user_messages=user_messages, embeddings=embeddings)
        payload = dict(state)
        payload["session_id"] = str(session_id)
        return payload

    def assign_topics(
        self,
        *,
        user_messages: list[str],
        embeddings: list[list[float]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        return self._algo.assign_topics(user_messages=user_messages, embeddings=embeddings)

    def update_with_current_user_message(
        self,
        *,
        session_id: str,
        state: Optional[dict[str, Any]],
        user_message: str,
        embedding: list[float],
    ) -> tuple[dict[str, Any], TopicSelection]:
        if not self._settings.enabled:
            selection = TopicSelection(topic_id="topic_0", action="same_topic", similarity=1.0)
            new_state = {"v": self._VERSION, "session_id": str(session_id), "current_topic_id": selection.topic_id, "topics": []}
            return new_state, selection

        new_state, selection = self._algo.update_with_current_user_message(
            state=state,
            user_message=user_message,
            embedding=embedding,
        )
        payload = dict(new_state)
        payload["session_id"] = str(session_id)
        self.save(session_id, payload)
        return payload, selection


__all__ = ["TopicStackManager", "TopicStackSettings", "TopicSelection"]

