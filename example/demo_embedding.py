import json
from dotenv import load_dotenv
load_dotenv()
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig

JSON_CONFIG = """
{
    "type": "qwen_embedding",
    "loading_method": "huggingface",
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cuda:0",
    "cache_folder": "./models/all-MiniLM-L6-v2",
    "use_china_mirror": true,
    "encode_kwargs": {
        "batch_size": 1,
        "show_progress_bar": false
    }
}
"""


def main() -> None:
    """使用 JSON 配置构建 QwenEmbeddingLLM，并对文本生成向量。"""
    config_data = json.loads(JSON_CONFIG)
    embedding_config = QwenEmbeddingConfig.model_validate(config_data)
    embedding_model = embedding_config.build()

    texts = [
        "OpenAI 于 2015 年成立。",
        "萨姆·阿尔特曼是 OpenAI 的 CEO。"
    ]

    embeddings = embedding_model.embed(texts)

    for text, vector in zip(texts, embeddings):
        print(f"文本: {text}")
        print(f"向量维度: {len(vector)}")
        print("---")


if __name__ == "__main__":
    main()

