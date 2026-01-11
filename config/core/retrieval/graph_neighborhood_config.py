from typing import Annotated, Literal, Optional, Union, List

from pydantic import Field

from framework.config import AbstractConfig
from config.encapsulation.database.graph_db.networkx_config import NetworkXConfig
from config.encapsulation.database.graph_db.networkx_with_embedding_config import NetworkXVectorConfig
from core.retrieval.graph_neighborhood_retriever import GraphNeighborhoodRetriever


class GraphNeighborhoodRetrieverConfig(AbstractConfig):
    type: Literal["graph_neighborhood"] = "graph_neighborhood"

    graph_store_config: Annotated[Union[NetworkXVectorConfig, NetworkXConfig], Field(discriminator="type")] = Field(
        description="NetworkX graph store config (must be local).",
    )

    stopwords_language: str = Field(default="en")
    extra_stopwords: Optional[List[str]] = Field(default=None)

    max_entity_words: int = Field(default=4, ge=1)
    min_entity_len_chars: int = Field(default=3, ge=1)
    max_seed_entities: int = Field(default=8, ge=1)

    enable_neighbor_expansion: bool = Field(default=True)
    neighbor_traversal_mode: Literal["both", "out", "in"] = Field(
        default="both",
        description="Which direction(s) to traverse when expanding entity neighbors.",
    )
    max_neighbors_per_seed: int = Field(default=8, ge=0)
    max_chunks_per_entity: int = Field(default=50, ge=1)
    allowed_relation_types: Optional[List[str]] = Field(
        default=None,
        description="If provided, only traverse entity-entity edges whose relation_type is in this allowlist.",
    )

    enable_fuzzy_entity_match: bool = Field(
        default=False,
        description="Best-effort substring matching when no exact entity key hit is found.",
    )
    max_fuzzy_matches_per_seed: int = Field(default=4, ge=0)

    max_mentions_per_seed_entity: int = Field(
        default=0,
        ge=0,
        description="If >0, skip seed entities whose mention chunk count exceeds this threshold (ambiguity control).",
    )

    seed_text_match_boost: float = Field(
        default=0.0,
        ge=0.0,
        description="Extra score per seed entity substring matched in the chunk text (precision proxy).",
    )

    seed_chunk_weight: float = Field(default=1.0)
    neighbor_chunk_weight: float = Field(default=0.3)

    def build(self):
        return GraphNeighborhoodRetriever(self)
