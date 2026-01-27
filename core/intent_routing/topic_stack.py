from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a @ b) / denom)


@dataclass(frozen=True)
class TopicStackSettings:
    max_topics: int
    same_topic_threshold: float
    return_to_topic_threshold: float
    sample_keep: int = 3


@dataclass(frozen=True)
class TopicSelection:
    topic_id: str
    action: str  # "same_topic" | "topic_switch" | "return_to_topic"
    similarity: float


class TopicStack:
    """Pure topic-stack state machine (session scoped; persistence is handled externally)."""

    VERSION = 1

    def __init__(self, *, settings: TopicStackSettings) -> None:
        self._settings = settings

    def rebuild_from_user_messages(self, *, user_messages: list[str], embeddings: list[list[float]]) -> dict[str, Any]:
        """Rebuild topic stack state from the latest user messages (chronological)."""
        topics: list[dict[str, Any]] = []
        current_topic_id: str | None = None

        for text, emb in zip(user_messages, embeddings):
            selection = self._select_topic_for_vector(topics=topics, vector=np.array(emb, dtype=np.float32))
            current_topic_id = selection.topic_id
            topics = self._apply_update(topics=topics, selection=selection, vector=np.array(emb, dtype=np.float32), text=str(text))

        return {
            "v": self.VERSION,
            "current_topic_id": current_topic_id,
            "topics": topics,
        }

    def assign_topics(
        self,
        *,
        user_messages: list[str],
        embeddings: list[list[float]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Assign each user message to a topic id (derived; chronological order)."""
        topics: list[dict[str, Any]] = []
        assigned: list[str] = []
        for text, emb in zip(user_messages, embeddings):
            selection = self._select_topic_for_vector(topics=topics, vector=np.array(emb, dtype=np.float32))
            assigned.append(selection.topic_id)
            topics = self._apply_update(
                topics=topics,
                selection=selection,
                vector=np.array(emb, dtype=np.float32),
                text=str(text),
            )
        return topics, assigned

    def update_with_current_user_message(
        self,
        *,
        state: Optional[dict[str, Any]],
        user_message: str,
        embedding: list[float],
    ) -> tuple[dict[str, Any], TopicSelection]:
        topics: list[dict[str, Any]] = []
        if isinstance(state, dict):
            topics = state.get("topics") if isinstance(state.get("topics"), list) else []
        vector = np.array(embedding, dtype=np.float32)
        selection = self._select_topic_for_vector(topics=topics, vector=vector)
        topics = self._apply_update(topics=topics, selection=selection, vector=vector, text=str(user_message))
        new_state = {
            "v": self.VERSION,
            "current_topic_id": selection.topic_id,
            "topics": topics,
        }
        return new_state, selection

    def _select_topic_for_vector(self, *, topics: list[dict[str, Any]], vector: np.ndarray) -> TopicSelection:
        if not topics:
            return TopicSelection(topic_id="topic_0", action="topic_switch", similarity=0.0)

        # current topic is always topics[0]
        current = topics[0]
        cur_centroid = np.array(current.get("centroid") or [], dtype=np.float32)
        if cur_centroid.size == vector.size and cur_centroid.size > 0:
            cur_sim = _cosine(vector, cur_centroid)
            if cur_sim >= float(self._settings.same_topic_threshold):
                return TopicSelection(topic_id=str(current.get("topic_id") or "topic_0"), action="same_topic", similarity=cur_sim)

        # Check if it matches any previous topic strongly enough (return-to-topic).
        best_sim = -1.0
        best_idx = -1
        for i, t in enumerate(topics):
            centroid = np.array(t.get("centroid") or [], dtype=np.float32)
            if centroid.size != vector.size or centroid.size == 0:
                continue
            sim = _cosine(vector, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0 and best_sim >= float(self._settings.return_to_topic_threshold):
            return TopicSelection(topic_id=str(topics[best_idx].get("topic_id") or f"topic_{best_idx}"), action="return_to_topic", similarity=float(best_sim))

        # Otherwise start a new topic.
        next_id = f"topic_{len(topics)}"
        return TopicSelection(topic_id=next_id, action="topic_switch", similarity=0.0)

    def _apply_update(
        self,
        *,
        topics: list[dict[str, Any]],
        selection: TopicSelection,
        vector: np.ndarray,
        text: str,
    ) -> list[dict[str, Any]]:
        topics = list(topics or [])

        def _bump_samples(existing: list[str], new_text: str) -> list[str]:
            keep = max(int(self._settings.sample_keep), 0)
            samples = [s for s in existing if isinstance(s, str) and s.strip()]
            if new_text.strip():
                samples.append(new_text.strip())
            if keep == 0:
                return []
            if len(samples) > keep:
                samples = samples[-keep:]
            return samples

        if selection.action == "topic_switch" and not topics:
            return [
                {
                    "topic_id": selection.topic_id,
                    "centroid": vector.tolist(),
                    "n": 1,
                    "samples": _bump_samples([], text),
                }
            ]

        # Find or create the topic.
        idx = next((i for i, t in enumerate(topics) if str(t.get("topic_id")) == selection.topic_id), -1)
        if idx < 0:
            topics.append({"topic_id": selection.topic_id, "centroid": vector.tolist(), "n": 1, "samples": _bump_samples([], text)})
            idx = len(topics) - 1

        # Update centroid with incremental mean.
        t = dict(topics[idx])
        centroid = np.array(t.get("centroid") or [], dtype=np.float32)
        n = int(t.get("n") or 0)
        if centroid.size != vector.size or centroid.size == 0 or n <= 0:
            centroid = vector
            n = 1
        else:
            centroid = (centroid * n + vector) / float(n + 1)
            n += 1
        t["centroid"] = centroid.tolist()
        t["n"] = n
        t["samples"] = _bump_samples(t.get("samples") if isinstance(t.get("samples"), list) else [], text)
        topics[idx] = t

        # Move selected topic to the front (stack top).
        if idx != 0:
            topic_obj = topics.pop(idx)
            topics.insert(0, topic_obj)

        # Cap topics.
        max_topics = max(int(self._settings.max_topics), 1)
        if len(topics) > max_topics:
            topics = topics[:max_topics]
        return topics
