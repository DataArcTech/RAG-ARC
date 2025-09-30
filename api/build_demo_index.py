import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encapsulation.data_model.schema import Chunk
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig

def load_real_data():
    """加载数据文件"""
    import json

    data_file = "./test/test.json"
    print(f"1. 加载数据文件: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = []
    for i, item in enumerate(data):
        # 创建Chunk ID
        chunk_id = item.get('id', f"chunk_{i}")

        # 获取内容
        content = item.get('content', '')

        # 创建元数据
        metadata = item.get('metadata', {})

        # 创建Chunk对象
        chunk = Chunk(
            id=chunk_id,
            content=content,
            metadata=metadata
        )
        chunks.append(chunk)

    print(f"成功加载 {len(chunks)} 个Chunk")
    return chunks


def build_demo_index():
    """构建演示用的FAISS和BM25索引"""
    print("=== 构建演示索引 ===")


    # 1. 加载数据
    chunks = load_real_data()

    # 2. 创建FAISS索引配置
    print("\n2. 创建FAISS索引配置...")

    embedding_config = QwenEmbeddingConfig(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="cuda:0", 
        encode_kwargs={
            "batch_size": 32,
            "show_progress_bar": True,
            # "multi_process": True
        },
        use_china_mirror=True,
        cache_folder="./models/Qwen"
    )

    # FAISS索引配置
    faiss_config = FaissVectorDBConfig(
        index_path="./data/unified_faiss_index",
        metric="cosine",
        index_type="flat",
        normalize_L2=True,
        embedding_config=embedding_config
    )

    print("FAISS配置创建完成")

    # 3. 构建FAISS索引
    print("\n3. 构建FAISS索引...")
    faiss_index = faiss_config.build()
    faiss_index.build_index(chunks)
    print("FAISS索引构建完成")

    # 4. 保存FAISS索引
    print("\n4. 保存FAISS索引...")
    os.makedirs("./data/unified_faiss_index", exist_ok=True)
    faiss_index.save_index("./data/unified_faiss_index", "index")
    print("FAISS索引已保存到 ./data/unified_faiss_index")

    # 5. 验证FAISS索引
    print("\n5. 验证FAISS索引...")
    info = faiss_index.get_vector_db_info()
    print(info)

    # 6. 创建BM25索引配置
    print("\n6. 创建BM25索引配置...")
    from config.encapsulation.database.bm25_config import BM25BuilderConfig

    bm25_config = BM25BuilderConfig(
        index_path="./data/unified_bm25_index",
        bm25_k1=1.2,
        bm25_b=0.75
    )

    print("BM25配置创建完成")

    # 7. 构建BM25索引
    print("\n7. 构建BM25索引...")

    # 先删除现有索引（如果存在）
    import shutil
    bm25_index_path = "./data/unified_bm25_index"
    if os.path.exists(bm25_index_path):
        print(f"删除现有BM25索引: {bm25_index_path}")
        shutil.rmtree(bm25_index_path)

    print(f"准备构建BM25索引，Chunk数量: {len(chunks)}")
    bm25_index = bm25_config.build()

    try:
        bm25_index.build_index(chunks)
        print("✓ BM25索引构建完成")
    except Exception as e:
        print(f"✗ BM25索引构建失败: {e}")
        # 尝试使用add_chunks方法
        try:
            print("尝试使用add_chunks方法...")
            bm25_index.update_index(chunks)
            print("✓ 使用add_chunks方法成功")
        except Exception as e2:
            print(f"✗ add_chunks也失败: {e2}")
            raise e

    # 8. 保存BM25索引
    print("\n8. 保存BM25索引...")
    os.makedirs("./data/unified_bm25_index", exist_ok=True)
    bm25_index.save_index("./data/unified_bm25_index", "index")
    print("BM25索引已保存到 ./data/unified_bm25_index")

    # 9. 验证BM25索引
    print("\n9. 验证BM25索引...")
    bm25_info = bm25_index.get_vector_db_info()
    print(bm25_info)

    print("\n所有demo索引构建完成！")
    print("现在可以运行 python api/quick_start.py 来使用预构建的索引")
    print(f"总共处理了 {len(chunks)} 个Chunk")
        

if __name__ == "__main__":
    build_demo_index()
