from typing import List

from encapsulation.data_model.schema import Chunk
from framework.module import AbstractModule


class NoOpReranker(AbstractModule):
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        return list(chunks or [])

