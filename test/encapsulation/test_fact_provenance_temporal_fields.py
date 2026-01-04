from encapsulation.database.utils.fact_provenance import upsert_fact_occurrence


def test_fact_provenance_keeps_latest_temporal_fields() -> None:
    by_id: dict = {}

    upsert_fact_occurrence(
        by_id,
        head_id="entity-a",
        head_name="A",
        relation_type="HAS_POLICY",
        tail_id="entity-b",
        tail_name="0天/周",
        chunk_id="chunk-1",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
        valid_from="2024-01-01T00:00:00+00:00",
        effective_date="2024-01-01T00:00:00+00:00",
        valid_to=None,
    )
    fact_id = next(iter(by_id.keys()))
    assert by_id[fact_id]["valid_from"] == "2024-01-01T00:00:00+00:00"
    assert by_id[fact_id]["effective_date"] == "2024-01-01T00:00:00+00:00"
    assert by_id[fact_id]["valid_to"] is None

    # Later evidence updates the effective date (take max ISO string).
    upsert_fact_occurrence(
        by_id,
        head_id="entity-a",
        head_name="A",
        relation_type="HAS_POLICY",
        tail_id="entity-b",
        tail_name="0天/周",
        chunk_id="chunk-2",
        owner_id="owner-1",
        db_owner_id="owner-1",
        schema_version="v1",
        domain="default",
        valid_from="2024-06-01T00:00:00+00:00",
        effective_date="2024-06-01T00:00:00+00:00",
        valid_to="2024-12-31T00:00:00+00:00",
    )
    assert by_id[fact_id]["valid_from"] == "2024-06-01T00:00:00+00:00"
    assert by_id[fact_id]["effective_date"] == "2024-06-01T00:00:00+00:00"
    assert by_id[fact_id]["valid_to"] == "2024-12-31T00:00:00+00:00"
