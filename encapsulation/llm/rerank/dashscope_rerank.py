"""DashScope-compatible rerank API client.

Calls the DashScope ``/reranks`` endpoint (qwen3-rerank or compatible models)
and returns ranked results in the standard ``List[Tuple[int, float]]`` format
expected by the RAG-ARC rerank interface.

Both synchronous (``rerank``) and asynchronous (``arerank``) methods are
provided so the client can be used from either context.
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

from .base import RerankLLMBase

if TYPE_CHECKING:
    from encapsulation.data_model.schema import Chunk

logger = logging.getLogger(__name__)

# Default DashScope rerank instruction (mirrors the official example).
_DEFAULT_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query."
)


class DashScopeRerankLLM(RerankLLMBase):
    """DashScope (qwen3-rerank) API-based reranker.

    This implementation calls the DashScope-compatible ``/reranks`` REST
    endpoint and is designed for *fast* file-candidate reranking inside
    DeepSearch's ``locate`` tool where latency is critical.

    Parameters are read from the config object (a ``DashScopeRerankConfig``
    instance) which in turn sources defaults from environment variables.
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.api_key: str = getattr(config, "api_key", "") or ""
        self.base_url: str = (
            getattr(config, "base_url", "") or "https://dashscope.aliyuncs.com/compatible-api/v1"
        ).rstrip("/")
        self.model_name: str = getattr(config, "model_name", "") or "qwen3-rerank"
        self.instruct: str = getattr(config, "instruct", "") or _DEFAULT_INSTRUCT
        self.timeout_seconds: float = float(getattr(config, "timeout_seconds", 30.0) or 30.0)

    # ------------------------------------------------------------------
    # Public: synchronous rerank (RerankLLMBase interface)
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List["Chunk"],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Synchronous rerank via DashScope API."""
        if not chunks:
            return []
        documents = [str(c.content or "") for c in chunks]
        return self._call_sync(query=query, documents=documents, top_k=top_k)

    # ------------------------------------------------------------------
    # Public: async rerank (used by DeepSearch locate)
    # ------------------------------------------------------------------

    async def arerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Asynchronous rerank via DashScope API."""
        if not documents:
            return []
        return await self._call_async(query=query, documents=documents, top_k=top_k)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(
        self, query: str, documents: List[str], top_k: Optional[int]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "instruct": self.instruct,
        }
        if top_k is not None and int(top_k) > 0:
            payload["top_n"] = int(top_k)
        return payload

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_response(data: Dict[str, Any], num_documents: int) -> List[Tuple[int, float]]:
        """Parse the DashScope rerank response into ``(index, score)`` tuples."""
        results = data.get("results") or []
        if not isinstance(results, list):
            logger.warning("DashScope rerank response missing 'results' key; returning empty.")
            return []
        parsed: List[Tuple[int, float]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            score = item.get("relevance_score")
            if idx is None or score is None:
                continue
            try:
                idx_i = int(idx)
                score_f = float(score)
            except (TypeError, ValueError):
                continue
            if 0 <= idx_i < num_documents:
                parsed.append((idx_i, score_f))
        # Sort descending by score (the API usually returns sorted, but be safe).
        parsed.sort(key=lambda t: t[1], reverse=True)
        return parsed

    def _call_sync(
        self, *, query: str, documents: List[str], top_k: Optional[int]
    ) -> List[Tuple[int, float]]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "httpx is required for DashScope rerank; install the 'dev' extras."
            ) from exc

        url = f"{self.base_url}/reranks"
        payload = self._build_payload(query, documents, top_k)
        headers = self._build_headers()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("DashScope rerank sync call failed: %s", exc)
            raise RuntimeError(f"DashScope rerank failed: {exc}") from exc

        if not isinstance(data, dict):
            return []
        return self._parse_response(data, len(documents))

    async def _call_async(
        self, *, query: str, documents: List[str], top_k: Optional[int]
    ) -> List[Tuple[int, float]]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "httpx is required for DashScope rerank; install the 'dev' extras."
            ) from exc

        url = f"{self.base_url}/reranks"
        payload = self._build_payload(query, documents, top_k)
        headers = self._build_headers()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error("DashScope rerank async call failed: %s", exc)
            raise RuntimeError(f"DashScope rerank failed: {exc}") from exc

        if not isinstance(data, dict):
            return []
        return self._parse_response(data, len(documents))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return ``True`` when an API key has been provided."""
        return bool(self.api_key)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "base_url": self.base_url,
            "provider": "dashscope",
            "class_name": self.__class__.__name__,
            "config_type": getattr(self.config, "type", "unknown"),
        }
