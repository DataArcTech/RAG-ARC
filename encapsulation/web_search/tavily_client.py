from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


@dataclass(frozen=True)
class TavilySearchResult:
    title: str
    url: str | None
    content: str
    score: float | None = None


class TavilySearchClient:
    """Minimal Tavily client used by HippoRAG Q&A and DeepSearch external channel."""

    def __init__(
        self,
        *,
        api_key: Optional[str],
        endpoint_url: str = "https://api.tavily.com/search",
        timeout_seconds: float = 20.0,
        search_depth: Literal["basic", "advanced"] = "advanced",
        max_results: int = 5,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._endpoint_url = str(endpoint_url or "").strip() or "https://api.tavily.com/search"
        self._timeout_seconds = float(timeout_seconds)
        self._search_depth = search_depth
        self._max_results = int(max_results)

    def is_configured(self) -> bool:
        return bool(self._api_key)

    @staticmethod
    def _coerce_results(payload: Dict[str, Any]) -> List[TavilySearchResult]:
        results = payload.get("results") or []
        if not isinstance(results, list):
            return []
        normalized: List[TavilySearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            normalized.append(
                TavilySearchResult(
                    title=str(item.get("title") or "Result").strip() or "Result",
                    url=(str(item.get("url") or "").strip() or None),
                    content=content,
                    score=(float(item["score"]) if isinstance(item.get("score"), (int, float)) else None),
                )
            )
        return normalized

    def search(self, *, query: str, max_results: Optional[int] = None) -> List[TavilySearchResult]:
        api_key = self._api_key
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")
        max_results = int(max_results) if max_results is not None else self._max_results
        max_results = max(1, max_results)
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("httpx is required for Tavily search; install the 'dev' extras.") from exc

        payload = {
            "api_key": api_key,
            "query": str(query or "").strip(),
            "search_depth": self._search_depth,
            "max_results": max_results,
        }
        with httpx.Client(timeout=self._timeout_seconds) as client:
            response = client.post(self._endpoint_url, json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            return []
        return self._coerce_results(data)

    async def asearch(self, *, query: str, max_results: Optional[int] = None) -> List[TavilySearchResult]:
        api_key = self._api_key
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is not configured.")
        max_results = int(max_results) if max_results is not None else self._max_results
        max_results = max(1, max_results)
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("httpx is required for Tavily search; install the 'dev' extras.") from exc

        payload = {
            "api_key": api_key,
            "query": str(query or "").strip(),
            "search_depth": self._search_depth,
            "max_results": max_results,
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(self._endpoint_url, json=payload)
            response.raise_for_status()
            data = response.json()
        if not isinstance(data, dict):
            return []
        return self._coerce_results(data)

    @staticmethod
    def to_evidence_chunks(
        *,
        results: List[TavilySearchResult],
        step_id: str,
        query: str,
        source: str = "web.tavily",
    ) -> List[Dict[str, Any]]:
        evidences: List[Dict[str, Any]] = []
        for idx, item in enumerate(results):
            evidences.append(
                {
                    "chunk_id": f"tavily-{step_id}-{idx}",
                    "source": source,
                    "content": f"{item.title}\n{item.content}",
                    "score": item.score,
                    "provenance": {
                        "provider": "tavily",
                        "url": item.url,
                        "step_id": step_id,
                        "query": query,
                        "snippet_rank": idx + 1,
                    },
                }
            )
        return evidences

