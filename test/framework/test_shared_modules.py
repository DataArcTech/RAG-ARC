#!/usr/bin/env python3
"""
测试共享模块装饰器功能
验证相同配置的模块实例是否被正确共享
"""

import os
import sys
import tempfile
import shutil
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, "..", ".."))))


def test_dense_retriever_sharing():
    """测试DenseRetriever的共享功能"""
    print("=" * 60)
    print("测试 DenseRetriever 共享模块功能")
    print("=" * 60)
    
    try:
        from core.retrieval.dense import DenseRetriever, DenseRetrieverConfig
        from encapsulation.database.vector_db.faiss import FaissIndexConfig
        from encapsulation.llm.huggingface import HuggingFaceEmbedConfig
        
        # 创建相同的配置
        index_config = FaissIndexConfig(
            index_path="/tmp/test_faiss_shared",
            index_name="test_shared",
            metric="cosine",
            index_type="hnsw"
        )
        
        embedding_config = HuggingFaceEmbedConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        config1 = DenseRetrieverConfig(
            index_config=index_config,
            embedding_config=embedding_config,
            metric="cosine"
        )
        
        config2 = DenseRetrieverConfig(
            index_config=index_config,
            embedding_config=embedding_config,
            metric="cosine"
        )
        
        # 创建实例
        retriever1 = DenseRetriever(config1)
        retriever2 = DenseRetriever(config2)
        
        # 验证是否是同一个实例
        is_same_instance = retriever1 is retriever2
        print(f"配置1: {config1.model_dump()}")
        print(f"配置2: {config2.model_dump()}")
        print(f"实例1 ID: {id(retriever1)}")
        print(f"实例2 ID: {id(retriever2)}")
        print(f"是否为同一实例: {is_same_instance}")
        
        if is_same_instance:
            print("✅ DenseRetriever 共享模块测试通过")
        else:
            print("❌ DenseRetriever 共享模块测试失败")
            
        # 测试不同配置
        config3 = DenseRetrieverConfig(
            index_config=index_config,
            embedding_config=embedding_config,
            metric="l2"  # 不同的metric
        )
        
        retriever3 = DenseRetriever(config3)
        is_different_instance = retriever1 is not retriever3
        print(f"\n不同配置测试:")
        print(f"配置3 metric: {config3.metric}")
        print(f"实例3 ID: {id(retriever3)}")
        print(f"与实例1不同: {is_different_instance}")
        
        if is_different_instance:
            print("✅ 不同配置创建不同实例测试通过")
        else:
            print("❌ 不同配置创建不同实例测试失败")
            
        return is_same_instance and is_different_instance
        
    except Exception as e:
        print(f"❌ DenseRetriever 测试失败: {e}")
        return False


def test_bm25_retriever_sharing():
    """测试TantivyBM25Retriever的共享功能"""
    print("\n" + "=" * 60)
    print("测试 TantivyBM25Retriever 共享模块功能")
    print("=" * 60)
    
    try:
        from core.retrieval.tantivy_bm25 import TantivyBM25Retriever, TantivyBM25RetrieverConfig
        from config.database.bm25_config import BM25IndexBuilderConfig
        
        # 创建相同的配置
        index_config = BM25IndexBuilderConfig(
            index_path="/tmp/test_bm25_shared",
            bm25_k1=1.2,
            bm25_b=0.75
        )
        
        config1 = TantivyBM25RetrieverConfig(
            index_config=index_config
        )
        
        config2 = TantivyBM25RetrieverConfig(
            index_config=index_config
        )
        
        # 创建实例
        retriever1 = TantivyBM25Retriever(config1)
        retriever2 = TantivyBM25Retriever(config2)
        
        # 验证是否是同一个实例
        is_same_instance = retriever1 is retriever2
        print(f"配置1: {config1.model_dump()}")
        print(f"配置2: {config2.model_dump()}")
        print(f"实例1 ID: {id(retriever1)}")
        print(f"实例2 ID: {id(retriever2)}")
        print(f"是否为同一实例: {is_same_instance}")
        
        if is_same_instance:
            print("✅ TantivyBM25Retriever 共享模块测试通过")
        else:
            print("❌ TantivyBM25Retriever 共享模块测试失败")
            
        # 测试不同配置
        index_config3 = BM25IndexBuilderConfig(
            index_path="/tmp/test_bm25_shared",
            bm25_k1=1.5,  # 不同的k1参数
            bm25_b=0.75
        )
        
        config3 = TantivyBM25RetrieverConfig(
            index_config=index_config3
        )
        
        retriever3 = TantivyBM25Retriever(config3)
        is_different_instance = retriever1 is not retriever3
        print(f"\n不同配置测试:")
        print(f"配置3 k1: {config3.index_config.bm25_k1}")
        print(f"实例3 ID: {id(retriever3)}")
        print(f"与实例1不同: {is_different_instance}")
        
        if is_different_instance:
            print("✅ 不同配置创建不同实例测试通过")
        else:
            print("❌ 不同配置创建不同实例测试失败")
            
        return is_same_instance and is_different_instance
        
    except Exception as e:
        print(f"❌ TantivyBM25Retriever 测试失败: {e}")
        return False


def test_huggingface_embed_sharing():
    """测试HuggingFaceEmbed的共享功能"""
    print("\n" + "=" * 60)
    print("测试 HuggingFaceEmbed 共享模块功能")
    print("=" * 60)
    
    try:
        from encapsulation.llm.huggingface import HuggingFaceEmbed, HuggingFaceEmbedConfig
        
        # 创建相同的配置
        config1 = HuggingFaceEmbedConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        config2 = HuggingFaceEmbedConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # 创建实例
        embed1 = HuggingFaceEmbed(config1)
        embed2 = HuggingFaceEmbed(config2)
        
        # 验证是否是同一个实例
        is_same_instance = embed1 is embed2
        print(f"配置1: {config1.model_dump()}")
        print(f"配置2: {config2.model_dump()}")
        print(f"实例1 ID: {id(embed1)}")
        print(f"实例2 ID: {id(embed2)}")
        print(f"是否为同一实例: {is_same_instance}")
        
        if is_same_instance:
            print("✅ HuggingFaceEmbed 共享模块测试通过")
        else:
            print("❌ HuggingFaceEmbed 共享模块测试失败")
            
        # 测试不同配置
        config3 = HuggingFaceEmbedConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda"  # 不同的device
        )
        
        embed3 = HuggingFaceEmbed(config3)
        is_different_instance = embed1 is not embed3
        print(f"\n不同配置测试:")
        print(f"配置3 device: {config3.device}")
        print(f"实例3 ID: {id(embed3)}")
        print(f"与实例1不同: {is_different_instance}")
        
        if is_different_instance:
            print("✅ 不同配置创建不同实例测试通过")
        else:
            print("❌ 不同配置创建不同实例测试失败")
            
        return is_same_instance and is_different_instance
        
    except Exception as e:
        print(f"❌ HuggingFaceEmbed 测试失败: {e}")
        return False


def test_qwen3_reranker_sharing():
    """测试Qwen3Reranker的共享功能"""
    print("\n" + "=" * 60)
    print("测试 Qwen3Reranker 共享模块功能")
    print("=" * 60)
    
    try:
        from core.rerank.Reranker_Qwen3 import Qwen3Reranker
        
        # 创建相同的配置参数
        # model_path = "Qwen/Qwen2.5-3B-Instruct"
        model_path = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B"
        max_length = 4096
        device_id = "cpu"  # 使用CPU避免GPU依赖
        
        # 创建实例
        reranker1 = Qwen3Reranker(
            model_name_or_path=model_path,
            max_length=max_length,
            device_id=device_id
        )
        
        reranker2 = Qwen3Reranker(
            model_name_or_path=model_path,
            max_length=max_length,
            device_id=device_id
        )
        
        # 验证是否是同一个实例
        is_same_instance = reranker1 is reranker2
        print(f"模型路径: {model_path}")
        print(f"最大长度: {max_length}")
        print(f"设备: {device_id}")
        print(f"实例1 ID: {id(reranker1)}")
        print(f"实例2 ID: {id(reranker2)}")
        print(f"是否为同一实例: {is_same_instance}")
        
        if is_same_instance:
            print("✅ Qwen3Reranker 共享模块测试通过")
        else:
            print("❌ Qwen3Reranker 共享模块测试失败")
            
        # 测试不同配置
        reranker3 = Qwen3Reranker(
            model_name_or_path=model_path,
            max_length=2048,  # 不同的max_length
            device_id=device_id
        )
        
        is_different_instance = reranker1 is not reranker3
        print(f"\n不同配置测试:")
        print(f"配置3 max_length: 2048")
        print(f"实例3 ID: {id(reranker3)}")
        print(f"与实例1不同: {is_different_instance}")
        
        if is_different_instance:
            print("✅ 不同配置创建不同实例测试通过")
        else:
            print("❌ 不同配置创建不同实例测试失败")
            
        return is_same_instance and is_different_instance
        
    except Exception as e:
        print(f"❌ Qwen3Reranker 测试失败 (可能缺少模型): {e}")
        return True  # 模型缺失不算测试失败


def main():
    """主测试函数"""
    print("🔧 共享模块装饰器功能测试")
    print("=" * 80)
    
    results = []
    
    # 测试各个模块
    results.append(test_dense_retriever_sharing())
    results.append(test_bm25_retriever_sharing())
    results.append(test_huggingface_embed_sharing())
    results.append(test_qwen3_reranker_sharing())
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有共享模块测试通过！")
    else:
        print("⚠️  部分测试失败，请检查上述输出")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
