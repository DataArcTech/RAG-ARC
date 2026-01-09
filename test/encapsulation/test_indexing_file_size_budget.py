from pathlib import Path


def test_pruned_hipporag_neo4j_indexing_file_kept_under_budget() -> None:
    """Repository engineering constraint: avoid >1000-line mega-modules."""

    path = Path("encapsulation/database/graph_db/pruned_hipporag_neo4j_indexing.py")
    line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
    assert line_count <= 1000

