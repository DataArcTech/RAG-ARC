from config.encapsulation.database.graph_db.networkx_config import NetworkXConfig
from config.core.retrieval.graph_neighborhood_config import GraphNeighborhoodRetrieverConfig
from encapsulation.data_model.schema import Chunk, GraphData


def test_graph_neighborhood_retriever_trace_no_seed_entities(tmp_path) -> None:
    index_dir = tmp_path / "graph_index"
    store = NetworkXConfig(storage_path=str(index_dir), index_name="graph", unify_entities_by_name=True).build()
    store.add_chunk(Chunk(id="c1", content="Alice met Bob.", metadata={"source_file_id": "doc1"}))
    store.add_graph_data(
        GraphData(
            entities=[{"id": "e1", "entity_name": "Alice", "entity_type": "Person", "attributes": {}}],
            relations=[],
        ),
        "c1",
    )
    store.save_index(str(index_dir), "graph")

    cfg = GraphNeighborhoodRetrieverConfig(
        graph_store_config=NetworkXConfig(storage_path=str(index_dir), index_name="graph", unify_entities_by_name=True)
    )
    retriever = cfg.build()

    chunks, trace = retriever.retrieve_with_trace("no entities here", k=10)
    assert chunks == []
    assert trace.get("fallback_reason") == "no_seed_entities"

    chunks2, trace2 = retriever.retrieve_with_trace("Alice", k=10)
    assert len(chunks2) == 1
    assert trace2.get("seed_entities") and "alice" in trace2["seed_entities"]

