import json
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()
from config.encapsulation.database.graph_db.pruned_hipporag_igraph_config import (
    PrunedHippoRAGIGraphConfig,
)
from encapsulation.data_model.schema import Chunk, GraphData



GRAPH_CONFIG_JSON = """
{
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
}
"""


def build_sample_graph_data() -> Chunk:
    entities = [
        {"id": "e1", "entity_name": "OpenAI", "entity_type": "组织", "attributes": {}},
        {"id": "e2", "entity_name": "2015年", "entity_type": "日期", "attributes": {}},
        {"id": "e3", "entity_name": "萨姆·阿尔特曼", "entity_type": "人物", "attributes": {}},
        {"id": "e4", "entity_name": "CEO", "entity_type": "职位", "attributes": {}},
    ]

    relations = [
        ["OpenAI", "成立于", "2015年"],
        ["萨姆·阿尔特曼", "担任", "CEO"]
    ]

    graph_data = GraphData(entities=entities, relations=relations, metadata={"source": "demo"})
    chunk = Chunk(
        id="chunk-demo-1",
        content="OpenAI 于 2015 年成立，萨姆·阿尔特曼担任 CEO。",
        graph=graph_data,
    )
    return chunk


def main() -> None:
    """将抽取结果写入 PrunedHippoRAG 图谱存储。"""
    config_data = json.loads(GRAPH_CONFIG_JSON)
    graph_config = PrunedHippoRAGIGraphConfig.model_validate(config_data)
    graph_store = graph_config.build()

    chunk = build_sample_graph_data()

    # 直接在 Chunk 上附加抽取结果，然后使用 build_index 批量入库
    graph_store.build_index([chunk])
    graph_store.save_index(config_data['storage_path'])

    info = graph_store.get_graph_db_info()

    print("=== 图谱存储信息 ===")
    pprint(info)


if __name__ == "__main__":
    main()

