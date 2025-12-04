import asyncio
import json
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
from config.core.file_management.extractor.hipporag2_extractor_config import (
    HippoRAG2ExtractorConfig,
)
from encapsulation.data_model.schema import Chunk



JSON_CONFIG = """
{
    "type": "hipporag2_extractor",
    "llm_config": {
        "type": "openai_chat",
        "model_name": "gpt-4o-mini",
        "temperature": 0.0
    },
    "max_concurrent": 2
}
"""


async def main():
    """演示 HippoRAG2Extractor 并发处理多个 Chunk。"""
    config_data = json.loads(JSON_CONFIG)
    extractor_config = HippoRAG2ExtractorConfig.model_validate(config_data)
    extractor = extractor_config.build()

    chunks = [
        Chunk(
            content=(
                "OpenAI 是一家人工智能研究机构，成立于 2015 年。"
                "其目标是确保通用人工智能造福全人类。"
            )
        ),
        Chunk(
            content=(
                "萨姆·阿尔特曼是 OpenAI 的联合创始人兼 CEO，长期关注 AI 安全问题。"
            )
        ),
        Chunk(
            content=(
                "微软于 2019 年与 OpenAI 建立战略合作关系，双方在云算力上紧密协作。"
            )
        ),
    ]

    processed_chunks = await extractor(chunks)

    for idx, chunk in enumerate(processed_chunks, start=1):
        print(f"=== Chunk #{idx} ===")
        pprint(chunk.graph.entities)
        pprint(chunk.graph.relations)
        print()


if __name__ == "__main__":
    asyncio.run(main())

