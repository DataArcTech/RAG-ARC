from encapsulation.data_model.schema import Chunk, GraphData
from config.encapsulation.database.graph_db.networkx_config import NetworkXConfig


def test_networkx_graph_store_unify_entities_by_name() -> None:
    store = NetworkXConfig(storage_path=None, unify_entities_by_name=True).build()

    def add(cid: str) -> None:
        store.add_chunk(Chunk(id=cid, content=f"chunk {cid}", metadata={"source_file_id": "doc1"}))
        store.add_graph_data(
            GraphData(
                entities=[{"id": "e1", "entity_name": "Alice", "entity_type": "Person", "attributes": {}}],
                relations=[],
            ),
            cid,
        )

    add("c1")
    add("c2")

    alice_nodes = [
        node_id
        for node_id, data in store.graph.nodes(data=True)
        if data.get("node_type") == "Entity" and data.get("entity_name") == "Alice"
    ]
    assert len(alice_nodes) == 1

    g1 = store.get_graph_data("c1")
    g2 = store.get_graph_data("c2")
    assert any(e.get("entity_name") == "Alice" for e in g1.entities)
    assert any(e.get("entity_name") == "Alice" for e in g2.entities)

