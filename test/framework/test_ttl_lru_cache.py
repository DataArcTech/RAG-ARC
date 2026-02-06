from framework.cache import TTLRUCache


def test_ttl_lru_cache_basic_eviction():
    cache = TTLRUCache(max_entries=2, ttl_seconds=60.0)
    cache.set(("k", 1), "a")
    cache.set(("k", 2), "b")
    assert cache.get(("k", 1)) == "a"
    # Insert 3rd entry -> evict LRU (("k", 2)) because ("k", 1) was recently accessed.
    cache.set(("k", 3), "c")
    assert cache.get(("k", 2)) is None
    assert cache.get(("k", 1)) == "a"
    assert cache.get(("k", 3)) == "c"


def test_ttl_lru_cache_ttl_expiry(monkeypatch):
    # Avoid time.sleep in tests; patch monotonic.
    now = {"t": 100.0}

    import framework.cache.ttl_lru as ttl_lru_mod

    monkeypatch.setattr(ttl_lru_mod.time, "monotonic", lambda: now["t"])
    cache = TTLRUCache(max_entries=2, ttl_seconds=10.0)
    cache.set("a", 1)
    assert cache.get("a") == 1
    now["t"] = 111.0
    assert cache.get("a") is None

