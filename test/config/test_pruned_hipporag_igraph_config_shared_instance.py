import types

from config.encapsulation.database.graph_db.pruned_hipporag_igraph_config import PrunedHippoRAGIGraphConfig
from framework.shared_module_decorator import make_hashable


class _DummyStore:
    def __init__(self, config):  # noqa: ANN001
        self.config = config


def _make_cached_store_factory():
    instances = {}

    def getinstance(*, config):  # noqa: ANN001
        key = make_hashable(config.model_dump() or {})
        if key not in instances:
            instances[key] = _DummyStore(config=config)
        return instances[key]

    getinstance.__wrapped__ = _DummyStore
    return getinstance


def test_config_can_disable_shared_store_instance(monkeypatch):
    module = __import__(
        "config.encapsulation.database.graph_db.pruned_hipporag_igraph_config",
        fromlist=["PrunedHippoRAGIGraphStore"],
    )
    cached_factory = _make_cached_store_factory()
    monkeypatch.setattr(module, "PrunedHippoRAGIGraphStore", cached_factory)

    cfg_shared = PrunedHippoRAGIGraphConfig(
        embedding={"type": "qwen_embedding"},
        shared_instance=True,
    )
    a1 = cfg_shared.build()
    a2 = cfg_shared.build()
    assert a1 is a2

    cfg_isolated = cfg_shared.model_copy(update={"shared_instance": False})
    b1 = cfg_isolated.build()
    b2 = cfg_isolated.build()
    assert b1 is not b2

