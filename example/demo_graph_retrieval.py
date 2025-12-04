import json

from dotenv import load_dotenv
load_dotenv()
from config.core.retrieval.pruned_hipporag_config import PrunedHippoRAGRetrievalConfig



RETRIEVAL_CONFIG_JSON = """
{
    "type": "pruned_hipporag_retrieval",
    "graph_config": {
        "type": "pruned_hipporag_igraph",
        "storage_path": "./data/demo_graph_index",
        "add_synonymy_edges": true,
        "embedding": {
            "type": "qwen_embedding",
            "loading_method": "huggingface",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "cache_folder": "./models/all-MiniLM-L6-v2",
            "use_china_mirror": true,
            "encode_kwargs": {
                "batch_size": 1,
                "show_progress_bar": false
            }
        }
    },
    "enable_llm_reranking": true,
    "expansion_hops": 2,
    "include_chunk_neighbors": true,
    "enable_pruning": true,
    "fact_retrieval_top_k": 10,
    "max_facts_after_reranking": 3,
    "damping_factor": 0.5,
    "passage_node_weight": 0.05
}
"""


def main() -> None:
    retrieval_config_data = json.loads(RETRIEVAL_CONFIG_JSON)
    retrieval_config = PrunedHippoRAGRetrievalConfig.model_validate(retrieval_config_data)
    retriever = retrieval_config.build()

    query = "OpenAI 的创始人是谁？"
    results = retriever.retrieve(query, top_k=3)

    print(f"=== 查询: {query} ===")
    for idx, chunk in enumerate(results, start=1):
        print(f"结果 #{idx}")
        print(f"内容: {chunk.content}")


if __name__ == "__main__":
    main()

