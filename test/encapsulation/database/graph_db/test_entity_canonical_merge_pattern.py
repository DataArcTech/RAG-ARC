from pathlib import Path


def test_entity_canonical_merge_does_not_include_extra_labels() -> None:
    """
    Regression: MERGE must not include extra labels in the pattern when a uniqueness constraint
    exists on (:EntityCanonical {canonical_id}).

    If an older DB already has (:EntityCanonical {canonical_id}) without the extra label,
    `MERGE (n:EntityCanonical:Concept {canonical_id})` will attempt to create a new node and
    trigger Neo4j unique constraint violations.
    """

    repo_root = Path(__file__).resolve().parents[4]
    target = repo_root / "encapsulation/database/graph_db/pruned_hipporag_neo4j_indexing_ingest.py"
    content = target.read_text(encoding="utf-8", errors="ignore")

    # We allow the substring in explanatory comments, but the actual Cypher must not MERGE with extra labels.
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            continue
        assert not stripped.startswith("MERGE (n:EntityCanonical:Concept")
    assert "MERGE (n:EntityCanonical {canonical_id: c.canonical_id})" in content
    assert "SET n:Concept" in content
