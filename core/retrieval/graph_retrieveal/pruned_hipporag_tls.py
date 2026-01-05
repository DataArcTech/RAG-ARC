import logging
import threading
import uuid
from typing import Optional

logger = logging.getLogger(__name__)


class _PrunedHippoRAGTLSMixin:
    def _get_tls(self) -> threading.local:
        tls = getattr(self, "_tls", None)
        if tls is None:
            tls = threading.local()
            setattr(self, "_tls", tls)
        return tls

    def _tls_get_list(self, key: str) -> list:
        tls = self._get_tls()
        value = getattr(tls, key, None)
        if value is None:
            value = []
            setattr(tls, key, value)
        return value

    @property
    def passage_node_idxs(self) -> list[int]:
        return self._tls_get_list("passage_node_idxs")

    @passage_node_idxs.setter
    def passage_node_idxs(self, value: list[int]) -> None:
        setattr(self._get_tls(), "passage_node_idxs", value)

    @property
    def passage_node_keys(self) -> list[str]:
        return self._tls_get_list("passage_node_keys")

    @passage_node_keys.setter
    def passage_node_keys(self, value: list[str]) -> None:
        setattr(self._get_tls(), "passage_node_keys", value)

    def _build_node_mappings(self, owner_id: Optional[uuid.UUID] = None):
        """
        Build mappings between passage nodes and their indices in the graph.

        This creates two parallel lists:
        - passage_node_idxs: Graph indices for passage/chunk nodes
        - passage_node_keys: Chunk IDs corresponding to those indices

        Args:
            owner_id: Optional owner ID to filter chunks by ownership
        """
        self.passage_node_idxs = []
        self.passage_node_keys = []

        cursor = self.graph_store.conn.cursor()
        if owner_id:
            cursor.execute("SELECT chunk_id FROM chunks WHERE owner_id = ? ORDER BY ROWID", (str(owner_id),))
        else:
            cursor.execute("SELECT chunk_id FROM chunks ORDER BY ROWID")
        chunk_ids = [row[0] for row in cursor.fetchall()]

        for chunk_id in chunk_ids:
            if chunk_id in self.graph_store.node_to_idx:
                idx = self.graph_store.node_to_idx[chunk_id]
                self.passage_node_idxs.append(idx)
                self.passage_node_keys.append(chunk_id)

        logger.info(f"Built mappings for {len(self.passage_node_idxs)} passage nodes")

    @staticmethod
    def _owner_to_str(owner_id: Optional[uuid.UUID]) -> Optional[str]:
        return str(owner_id) if owner_id is not None else None

