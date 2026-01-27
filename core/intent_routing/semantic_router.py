from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import PrivateAttr
from semantic_router.encoders.base import DenseEncoder
from semantic_router.index.local import LocalIndex
from semantic_router.route import Route as SRRoute
from semantic_router.routers import SemanticRouter as SRSemanticRouter


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class IntentRoute:
    name: str
    action: Literal["rag", "web_only", "no_retrieval"]
    threshold: float
    utterances: list[str]


@dataclass(frozen=True)
class IntentMatch:
    intent: str
    action: Literal["rag", "web_only", "no_retrieval"]
    score: float
    utterance: str


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    action: Literal["rag", "web_only", "no_retrieval"]
    score: float
    # Optional observability payload. We keep it empty to avoid re-implementing semantic-router internals.
    top_matches: list[IntentMatch]


class _ProviderDenseEncoder(DenseEncoder):
    """DenseEncoder wrapper around our embedding provider.

    We keep intent embeddings decoupled from RAG embeddings by injecting a provider
    (local Qwen or OpenAI-compatible). This follows semantic-router's intended usage:
    the router calls the encoder for both route utterances and incoming queries.
    """

    _provider: EmbeddingProvider = PrivateAttr()

    def __init__(self, *, provider: EmbeddingProvider, name: str = "intent-embedder") -> None:
        super().__init__(name=name, score_threshold=None, type="custom")
        self._provider = provider

    def __call__(self, docs: list[object]) -> list[list[float]]:
        texts: list[str] = []
        for d in docs:
            if d is None:
                texts.append("")
            else:
                texts.append(str(d))
        return self._provider.embed(texts)

    async def acall(self, docs: list[object]) -> list[list[float]]:
        # Keep it simple: run sync embedding in async contexts too.
        return self.__call__(docs)


class SemanticIntentRouter:
    """Embedding-based intent router using the third-party `semantic-router` package.

    Notes:
    - We accept an injected embedding provider (local Qwen / OpenAI-compat) to keep
      intent embeddings decoupled from RAG embeddings.
    - Route scoring + thresholding is delegated to semantic-router.
    - `aggregation` and `top_k` are semantic-router-native knobs; expose them via config.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        routes: list[IntentRoute],
        default_intent: str,
        top_k: int = 5,
        aggregation: Literal["mean", "max", "sum"] = "mean",
    ) -> None:
        if not routes:
            raise ValueError("routes must be non-empty")
        self._default_intent = str(default_intent or "").strip()
        top_k = max(int(top_k), 1)

        route_by_name = {r.name: r for r in routes}
        if self._default_intent not in route_by_name:
            raise ValueError(f"default_intent {default_intent} not found in routes")
        self._route_by_name = route_by_name

        # Follow semantic-router best practice:
        # - provide a DenseEncoder and let the router embed/sync routes into the index
        # - use auto_sync="local" so local routes are the source of truth for the index
        self._sr = SRSemanticRouter(
            encoder=_ProviderDenseEncoder(provider=embedding_provider),
            routes=[
                SRRoute(
                    name=r.name,
                    utterances=list(r.utterances),
                    score_threshold=float(r.threshold),
                    metadata={"action": r.action},
                )
                for r in routes
            ],
            index=LocalIndex(),
            top_k=top_k,
            aggregation=aggregation,
            auto_sync="local",
        )

    def classify(self, query: str, *, route_filter: list[str] | None = None) -> IntentDecision:
        q = str(query or "").strip()
        if not q:
            r = self._route_by_name[self._default_intent]
            return IntentDecision(intent=r.name, action=r.action, score=0.0, top_matches=[])

        choice = self._sr(text=q, route_filter=route_filter)
        picked = str(getattr(choice, "name", "") or "").strip()
        score = float(getattr(choice, "similarity_score", 0.0) or 0.0)

        if not picked or picked not in self._route_by_name:
            r = self._route_by_name[self._default_intent]
            return IntentDecision(intent=r.name, action=r.action, score=0.0, top_matches=[])

        r = self._route_by_name[picked]
        return IntentDecision(intent=r.name, action=r.action, score=score, top_matches=[])
