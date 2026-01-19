from core.utils.quota import allocate_quotas


def test_allocate_quotas_basic_ratio() -> None:
    # 1 : 1 : 1.5 over k=30 -> 9, 9, 12 (with 1-each floor)
    out = allocate_quotas(30, [1.0, 1.0, 1.5])
    assert out.quotas == [9, 9, 12]
    assert sum(out.quotas) == 30


def test_allocate_quotas_graph_downweighted() -> None:
    # 1 : 1 : 0.5 over k=30 -> 12, 12, 6 (with 1-each floor)
    out = allocate_quotas(30, [1.0, 1.0, 0.5])
    assert out.quotas == [12, 12, 6]
    assert sum(out.quotas) == 30


def test_allocate_quotas_disable_graph() -> None:
    out = allocate_quotas(30, [1.0, 1.0, 0.0])
    assert out.quotas[2] == 0
    assert sum(out.quotas) == 30


def test_allocate_quotas_small_budget_floor() -> None:
    out = allocate_quotas(3, [1.0, 1.0, 1.5])
    # With k==active backends, each gets 1.
    assert out.quotas == [1, 1, 1]

