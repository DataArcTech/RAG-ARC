from typing import Literal

from framework.config import AbstractConfig
from core.rerank.noop_reranker import NoOpReranker


class NoOpRerankerConfig(AbstractConfig):
    type: Literal["noop_reranker"] = "noop_reranker"

    def build(self) -> NoOpReranker:
        return NoOpReranker(self)

