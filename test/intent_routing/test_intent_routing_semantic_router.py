from dataclasses import dataclass

from core.intent_routing.semantic_router import IntentRoute, SemanticIntentRouter


@dataclass
class _FakeEmbedder:
    """Deterministic tiny embedder for tests.

    Encodes texts into 3D vectors based on simple token presence.
    """

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            s = str(t or "").lower()
            # Keep tokens non-overlapping so tests remain deterministic across routers.
            if "web" in s or "online" in s or "网上" in s or "联网" in s:
                out.append([1.0, 0.0, 0.0])
            elif "hello" in s or "hi" in s or "thanks" in s or "thank" in s or "你好" in s or "谢谢" in s:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return out


def test_semantic_intent_router_selects_best_intent() -> None:
    router = SemanticIntentRouter(
        embedding_provider=_FakeEmbedder(),
        routes=[
            IntentRoute(
                name="RAG_REQUIRED",
                action="rag",
                threshold=0.6,
                utterances=["doc clause", "from the document"],
            ),
            IntentRoute(
                name="WEB_ONLY",
                action="web_only",
                threshold=0.6,
                utterances=["web search", "search online"],
            ),
            IntentRoute(
                name="NO_RETRIEVAL",
                action="no_retrieval",
                threshold=0.6,
                utterances=["hello", "thanks"],
            ),
        ],
        default_intent="RAG_REQUIRED",
        top_k=3,
    )

    assert router.classify("please do a web search").intent == "WEB_ONLY"
    assert router.classify("你好").intent == "NO_RETRIEVAL"
    assert router.classify("find doc clause").intent == "RAG_REQUIRED"


def test_semantic_intent_router_falls_back_when_below_threshold() -> None:
    router = SemanticIntentRouter(
        embedding_provider=_FakeEmbedder(),
        routes=[
            IntentRoute(
                name="RAG_REQUIRED",
                action="rag",
                threshold=1.01,  # too strict (cosine similarity <= 1.0)
                utterances=["doc clause"],
            ),
            IntentRoute(
                name="WEB_ONLY",
                action="web_only",
                threshold=1.01,  # too strict (cosine similarity <= 1.0)
                utterances=["web search"],
            ),
        ],
        default_intent="RAG_REQUIRED",
        top_k=2,
    )

    decision = router.classify("please do a web search")
    assert decision.intent == "RAG_REQUIRED"
    assert decision.action == "rag"


def test_semantic_intent_router_empty_query_is_deterministic() -> None:
    router = SemanticIntentRouter(
        embedding_provider=_FakeEmbedder(),
        routes=[
            IntentRoute(name="RAG_REQUIRED", action="rag", threshold=0.0, utterances=["doc clause"]),
        ],
        default_intent="RAG_REQUIRED",
        top_k=1,
    )
    decision = router.classify("")
    assert decision.intent == "RAG_REQUIRED"
    assert decision.action == "rag"

