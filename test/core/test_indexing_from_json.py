#!/usr/bin/env python3
"""
简单的索引测试代码
测试从JSON文件构建BM25索引
"""

import os
import tempfile
import shutil
import logging
import json
import asyncio
from typing import List
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config.encapsulation.database.bm25_config import BM25BuilderConfig
from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.core.file_management.indexing.faiss_indexing_config import FaissIndexerConfig


def create_test_json_files(temp_dir: str) -> List[str]:
    """创建测试JSON文件"""
    
    # 测试文档数据
    test_documents = [
        {
            "id": "doc1",
            "content": "人工智能是计算机科学的一个分支，它试图理解智能的实质。",
            "metadata": {"source": "AI教程", "category": "技术"}
        },
        {
            "id": "doc2",
            "content": "机器学习是人工智能的一个子领域，专注于算法和统计模型。",
            "metadata": {"source": "ML指南", "category": "技术"}
        },
        {
            "id": "doc3",
            "content": "深度学习使用神经网络来模拟人脑的工作方式。",
            "metadata": {"source": "DL手册", "category": "技术"}
        },
        {
            "id": "doc4",
            "content": "自然语言处理是人工智能的重要应用领域之一。",
            "metadata": {"source": "NLP概述", "category": "应用"}
        },
        {
            "id": "doc5",
            "content": "计算机视觉让机器能够理解和解释视觉信息。",
            "metadata": {"source": "CV基础", "category": "应用"}
        }
    ]
    
    # 创建JSON文件
    json_files = []
    for i, doc in enumerate(test_documents):
        file_path = os.path.join(temp_dir, f"doc_{i+1}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        json_files.append(file_path)
    
    return json_files


def test_bm25_indexing_from_json():
    """测试从JSON文件构建BM25索引"""
    print("\n=== 测试从JSON文件构建BM25索引 ===")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    bm25_index_path = os.path.join(temp_dir, "bm25_test")
    
    try:
        # 创建测试JSON文件
        json_files = create_test_json_files(temp_dir)
        print(f"创建了 {len(json_files)} 个JSON文件")
        for file_path in json_files:
            print(f"  - {os.path.basename(file_path)}")
        
        # 配置BM25构建器
        bm25_builder_config = BM25BuilderConfig(
            index_path=bm25_index_path,
            bm25_k1=1.2,
            bm25_b=0.75,
            batch_size=10,
            k=5,
            with_score=True
        )
        
        # 配置BM25索引器
        indexer_config = BM25IndexerConfig(
            index_config=bm25_builder_config
        )
        
        # 创建索引器
        indexer = indexer_config.build()
        print("BM25索引器创建成功")
        
        # 从JSON文件构建索引
        print("开始从JSON文件构建BM25索引...")
        
        async def build_index_from_files():
            success = await indexer.index_chunk_files(json_files)
            return success
        
        # 运行异步函数
        success = asyncio.run(build_index_from_files())
        
        if success:
            print("✅ 从JSON文件构建BM25索引成功")
        else:
            print("❌ 从JSON文件构建BM25索引失败")
            return False
        
        # 验证索引
        bm25_builder = indexer.bm25_builder
        info = bm25_builder.get_vector_db_info()
        print(f"BM25索引信息: {info}")
        
        # 测试检索文档
        retrieved_docs = bm25_builder.get_by_ids(["doc1", "doc2"])
        print(f"检索到 {len(retrieved_docs)} 个文档")
        for doc in retrieved_docs:
            print(f"  - {doc.id}: {doc.content[:50]}...")
        
        # 测试搜索功能
        search_results = bm25_builder.search("人工智能", k=3)
        print(f"搜索'人工智能'得到 {len(search_results)} 个结果")
        for doc in search_results:
            score = doc.metadata.get('score', 'N/A')
            print(f"  - {doc.id} (score: {score}): {doc.content[:50]}...")
        
        print("✅ BM25索引测试通过")
        return True
        
    except Exception as e:
        print(f"❌ BM25索引测试失败: {e}")
        logger.exception("BM25测试异常")
        return False
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def test_faiss_indexing_from_json():
    """测试从JSON文件构建FAISS索引"""
    print("\n=== 测试从JSON文件构建FAISS索引 ===")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    faiss_index_path = os.path.join(temp_dir, "faiss_test")

    try:
        # 创建测试JSON文件
        json_files = create_test_json_files(temp_dir)
        print(f"创建了 {len(json_files)} 个JSON文件")

        # 配置embedding模型
        embedding_config = QwenEmbeddingConfig(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device="cuda:0",
            use_china_mirror=True,
            cache_folder="./models/Qwen"
        )

        # 配置FAISS向量数据库
        faiss_config = FaissVectorDBConfig(
            type="faiss",
            index_path=faiss_index_path,
            metric="cosine",
            index_type="flat",
            normalize_L2=True,
            embedding_config=embedding_config
        )

        # 配置FAISS索引器
        indexer_config = FaissIndexerConfig(
            type="faiss_indexer",
            index_config=faiss_config
        )

        # 创建索引器
        indexer = indexer_config.build()
        print("FAISS索引器创建成功")

        # 从JSON文件构建索引
        print("开始从JSON文件构建FAISS索引...")

        async def build_index_from_files():
            success = await indexer.index_chunk_files(json_files)
            return success

        # 运行异步函数
        success = asyncio.run(build_index_from_files())

        if success:
            print("✅ 从JSON文件构建FAISS索引成功")
        else:
            print("❌ 从JSON文件构建FAISS索引失败")
            return False

        # 验证索引
        faiss_db = indexer.faiss_db
        info = faiss_db.get_vector_db_info()
        print(f"FAISS索引信息: {info}")

        # 测试检索文档
        retrieved_docs = faiss_db.get_by_ids(["doc1", "doc2"])
        print(f"检索到 {len(retrieved_docs)} 个文档")
        for doc in retrieved_docs:
            print(f"  - {doc.id}: {doc.content[:50]}...")

        print("✅ FAISS索引测试通过")
        return True

    except Exception as e:
        print(f"❌ FAISS索引测试失败: {e}")
        logger.exception("FAISS测试异常")
        return False
    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def main():
    """主测试函数"""
    print("开始从JSON文件构建索引测试...")

    # 测试BM25索引
    bm25_result = test_bm25_indexing_from_json()

    faiss_result = test_faiss_indexing_from_json()
    # 输出测试结果
    print("\n=== 测试结果汇总 ===")
    status = "✅ 通过" if bm25_result else "❌ 失败"
    print(f"BM25索引测试: {status}")
    status = "✅ 通过" if faiss_result else "❌ 失败"
    print(f"FAISS索引测试: {status}")

    return bm25_result


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
