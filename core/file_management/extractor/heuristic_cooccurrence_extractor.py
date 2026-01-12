import logging
from collections import Counter
from typing import TYPE_CHECKING

from core.file_management.extractor.base import ExtractorBase
from core.knowledge_graph.heuristic_entity import iter_candidate_phrases, normalize_entity_key
from core.utils.stopwords import get_stopwords
from encapsulation.data_model.schema import Chunk, GraphData

if TYPE_CHECKING:
    from config.core.file_management.extractor.heuristic_cooccurrence_extractor_config import (
        HeuristicCooccurrenceExtractorConfig,
    )

logger = logging.getLogger(__name__)


class HeuristicCooccurrenceExtractor(ExtractorBase):
    """LLM-free extractor for benchmark mode.

    This extractor builds a lightweight graph based on:
    - Heuristic entity phrase detection (proper-noun-like phrases)
    - Optional co-occurrence relations within the same sentence
    """

    def __init__(self, config: "HeuristicCooccurrenceExtractorConfig"):
        super().__init__(config)
        self._stopwords = get_stopwords(getattr(config, "stopwords_language", "en"))
        extra = getattr(config, "extra_stopwords", None) or []
        self._stopwords = self._stopwords.union({normalize_entity_key(x) for x in extra if str(x).strip()})

    async def extract(self, chunk: Chunk) -> GraphData:
        if not chunk.content:
            return GraphData()

        max_words = int(getattr(self.config, "max_entity_words", 4) or 4)
        min_chars = int(getattr(self.config, "min_entity_len_chars", 3) or 3)
        max_entities = int(getattr(self.config, "max_entities_per_chunk", 32) or 32)

        phrases = iter_candidate_phrases(chunk.content, max_words=max_words)
        key_counts: Counter[str] = Counter()
        key_to_surface: dict[str, str] = {}

        for phrase in phrases:
            key = normalize_entity_key(phrase)
            if not key or len(key) < min_chars:
                continue
            if key in self._stopwords:
                continue
            key_counts[key] += 1
            # Prefer longer surface forms for display.
            prev = key_to_surface.get(key)
            if prev is None or len(phrase) > len(prev):
                key_to_surface[key] = phrase

        if not key_counts:
            return GraphData()

        most_common = key_counts.most_common(max_entities)
        entities: list[dict] = []
        key_to_entity_name: dict[str, str] = {}
        for i, (key, _) in enumerate(most_common, start=1):
            name = key_to_surface.get(key) or key
            key_to_entity_name[key] = name
            entities.append(
                {
                    "id": f"e{i}",
                    "entity_name": name,
                    "entity_type": str(getattr(self.config, "default_entity_type", "Entity") or "Entity"),
                    "attributes": {},
                }
            )

        relations: list[list[str]] = []
        if bool(getattr(self.config, "enable_cooccurrence_relations", True)):
            relation_type = str(getattr(self.config, "cooccurrence_relation_type", "CO_OCCUR") or "CO_OCCUR")
            max_pairs = int(getattr(self.config, "max_cooccurrence_pairs_per_chunk", 256) or 256)
            keys = [key for key, _ in most_common]
            pair_count = 0
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    if pair_count >= max_pairs:
                        break
                    a = key_to_entity_name.get(keys[i])
                    b = key_to_entity_name.get(keys[j])
                    if not a or not b:
                        continue
                    relations.append([a, relation_type, b])
                    pair_count += 1
                if pair_count >= max_pairs:
                    break

        return GraphData(entities=entities, relations=relations, metadata={"extractor": "heuristic_cooccurrence"})
