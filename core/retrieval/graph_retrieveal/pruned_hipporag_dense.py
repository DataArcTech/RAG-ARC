import json
import logging
import uuid
from typing import List, Optional

import numpy as np

from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)


class _PrunedHippoRAGDenseMixin:
    def _dense_passage_retrieval_scores(self, query: str) -> np.ndarray:
        """
        Compute dense retrieval scores for all passages using dot product similarity.

        Args:
            query: Query string

        Returns:
            Array of similarity scores for all passages
        """
        query_embedding = self._get_query_embedding(query)

        # Collect passage embeddings
        passage_embeddings_list = []
        embedding_dim: int | None = None
        for chunk_id in self.passage_node_keys:
            if chunk_id in self.graph_store.chunk_embeddings:
                passage_embeddings_list.append(self.graph_store.chunk_embeddings[chunk_id])
            else:
                # Use zero embedding for missing chunks
                if embedding_dim is None:
                    if passage_embeddings_list:
                        embedding_dim = len(passage_embeddings_list[0])
                    else:
                        embedding_dim = self.graph_store.embedding_model.get_embedding_dimension()
                passage_embeddings_list.append(np.zeros(embedding_dim))

        if not passage_embeddings_list:
            logger.warning("No passage embeddings available")
            return np.zeros(len(self.passage_node_keys))

        passage_embeddings_array = np.array(passage_embeddings_list)

        # Compute dot product similarity
        query_doc_scores = np.dot(passage_embeddings_array, query_embedding)

        return query_doc_scores

    def _dense_passage_retrieval(
        self,
        query: str,
        top_k: int = 10,
        owner_id: Optional[uuid.UUID] = None,
        *,
        query_doc_scores: Optional[np.ndarray] = None,
    ) -> List[Chunk]:
        """
        Fallback dense retrieval method (used when graph retrieval fails).

        Args:
            query: Query string
            top_k: Number of chunks to retrieve
            owner_id: Optional owner ID to filter chunks

        Returns:
            List of retrieved Chunk objects
        """
        query_doc_scores = query_doc_scores if query_doc_scores is not None else self._dense_passage_retrieval_scores(query)

        top_k_indices = np.argsort(query_doc_scores)[-top_k:][::-1]

        chunk_ids = [self.passage_node_keys[i] for i in top_k_indices if i < len(self.passage_node_keys)]
        scores = [query_doc_scores[i] for i in top_k_indices if i < len(query_doc_scores)]

        chunks = self._convert_to_chunks(chunk_ids, scores, owner_id=owner_id)

        return chunks

    def _convert_to_chunks(self, chunk_ids: List[str], scores: List[float], owner_id: Optional[uuid.UUID] = None) -> List[Chunk]:
        """
        Convert chunk IDs and scores to Chunk objects by querying the database.

        Args:
            chunk_ids: List of chunk IDs
            scores: List of relevance scores
            owner_id: Optional owner ID to filter chunks

        Returns:
            List of Chunk objects with scores in metadata
        """
        chunks = []
        cursor = self.graph_store.conn.cursor()

        for chunk_id, score in zip(chunk_ids, scores):
            if owner_id is not None:
                cursor.execute(
                    "SELECT content, owner_id, metadata FROM chunks WHERE chunk_id = ? AND owner_id = ?",
                    (chunk_id, str(owner_id)),
                )
            else:
                cursor.execute(
                    "SELECT content, owner_id, metadata FROM chunks WHERE chunk_id = ?",
                    (chunk_id,),
                )

            row = cursor.fetchone()
            if row:
                content = row[0]
                chunk_owner_id = row[1]
                metadata = json.loads(row[2]) if row[2] else {}
                metadata["score"] = float(score)

                chunk = Chunk(
                    id=chunk_id,
                    content=content,
                    owner_id=chunk_owner_id,
                    metadata=metadata,
                )
                chunks.append(chunk)

        return chunks
