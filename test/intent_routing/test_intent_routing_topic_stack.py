from dataclasses import dataclass
from typing import Any, Optional

from application.intent_routing.topic_stack import TopicStackManager, TopicStackSettings


@dataclass
class _FakeRedis:
    store: dict[str, Any]

    def get(self, key: str, **kwargs: Any) -> Optional[Any]:
        return self.store.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None, **kwargs: Any) -> bool:
        self.store[key] = value
        return True


def test_topic_stack_return_to_previous_topic() -> None:
    mgr = TopicStackManager(
        redis=_FakeRedis(store={}),  # type: ignore[arg-type]
        settings=TopicStackSettings(
            enabled=True,
            redis_ttl_seconds=60,
            max_user_messages=10,
            max_topics=6,
            same_topic_threshold=0.82,
            return_to_topic_threshold=0.85,
            history_mode="topic_scoped",
            sample_keep=3,
        ),
    )

    user_messages = ["topic A", "topic B", "topic A again"]
    embeddings = [
        [1.0, 0.0],  # A
        [0.0, 1.0],  # B
        [1.0, 0.0],  # back to A
    ]

    topics, assigned = mgr.assign_topics(user_messages=user_messages, embeddings=embeddings)
    assert assigned[0] == "topic_0"
    assert assigned[1] == "topic_1"
    assert assigned[2] == "topic_0"
    assert topics and topics[0]["topic_id"] == "topic_0"

