import numpy as np

from encapsulation.database.graph_db.pruned_hipporag_neo4j_kg_maintenance import (
    _MentionRow,
    _PrunedHippoRAGNeo4jKGMaintenanceMixin,
)
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id


class _DummyStore(_PrunedHippoRAGNeo4jKGMaintenanceMixin):
    def __init__(self, *, chunk_embeddings: dict[str, np.ndarray]):
        self.chunk_embeddings = chunk_embeddings


def test_l1_build_payloads_clusters_mentions_and_emits_identities() -> None:
    owner_key = "owner-1"
    surface_entity_id = "ent-surface-1"

    # Two tight clusters far apart.
    chunk_embeddings = {
        "c1": np.asarray([1.0, 0.0], dtype=np.float32),
        "c2": np.asarray([0.99, 0.01], dtype=np.float32),
        "c3": np.asarray([-1.0, 0.0], dtype=np.float32),
        "c4": np.asarray([-0.99, -0.01], dtype=np.float32),
    }
    store = _DummyStore(chunk_embeddings=chunk_embeddings)

    rows = [
        _MentionRow(
            mention_id="m1",
            chunk_id="c1",
            surface_entity_id=surface_entity_id,
            source_file_id="f1",
            source_version="v1",
            valid_from="2020-01-01",
            valid_to="2020-12-31",
            effective_date="",
        ),
        _MentionRow(
            mention_id="m2",
            chunk_id="c2",
            surface_entity_id=surface_entity_id,
            source_file_id="f1",
            source_version="v1",
            valid_from="2019-01-01",
            valid_to="2021-01-01",
            effective_date="",
        ),
        _MentionRow(
            mention_id="m3",
            chunk_id="c3",
            surface_entity_id=surface_entity_id,
            source_file_id="f2",
            source_version="v2",
            valid_from="",
            valid_to="",
            effective_date="2022-06-01",
        ),
        _MentionRow(
            mention_id="m4",
            chunk_id="c4",
            surface_entity_id=surface_entity_id,
            source_file_id="f2",
            source_version="v2",
            valid_from="2022-01-01",
            valid_to="2023-01-01",
            effective_date="",
        ),
    ]

    by_surface = {surface_entity_id: rows}
    type_keys = {surface_entity_id: "product"}

    identities, assignments, mention_ids, identity_ids, missing = store._kg_l1_build_disambiguation_payloads(
        owner_key=owner_key,
        by_surface=by_surface,
        type_keys=type_keys,
        min_sim=0.8,
        min_mentions_to_split=0,
    )

    assert missing == 0
    assert sorted(mention_ids) == ["m1", "m2", "m3", "m4"]
    assert len(identities) == 2
    assert len(assignments) == 4

    # Identity IDs should be stable: surface + rep mention.
    rep_ids = sorted({i["identity_id"] for i in identities})
    expected = sorted(
        [
            compute_mdhash_id(f"{surface_entity_id}|m1", prefix="identity-", owner_id=owner_key),
            compute_mdhash_id(f"{surface_entity_id}|m3", prefix="identity-", owner_id=owner_key),
        ]
    )
    assert rep_ids == expected
    assert sorted(identity_ids) == expected

    # Time summary should be best-effort, string-based.
    by_id = {i["identity_id"]: i for i in identities}
    # Cluster containing m1+m2 should have min from 2019 and max to 2021.
    id_cluster_12 = compute_mdhash_id(f"{surface_entity_id}|m1", prefix="identity-", owner_id=owner_key)
    assert by_id[id_cluster_12]["valid_from_min"] == "2019-01-01"
    assert by_id[id_cluster_12]["valid_to_max"] == "2021-01-01"

    # Assignment confidence should be cosine sim in [-1, 1] (not a constant).
    confs = [a["confidence"] for a in assignments]
    assert all(isinstance(x, float) for x in confs)
    assert max(confs) <= 1.00001
    assert min(confs) >= -1.00001

