import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.retrieval_api import api
from core.utils.data_model import Document

def quick_start():
    """快速开始示例"""
    print("=== 统一检索API快速开始 ===")
    
    try:
        # 1. 从配置文件创建检索器（推荐方式）
        print("1. 创建检索器...")
        api.create_from_config_file(
            "demo_retriever", 
            "api/config_examples/unified_dense_config.json"
        )
        print("✓ 检索器创建成功")
        
        # 2. 添加示例文档
        print("\n2. 添加文档...")
        documents = [
            Document(id="1", content="人工智能是模拟人类智能的技术"),
            Document(id="2", content="机器学习是实现人工智能的重要方法"),
            Document(id="3", content="深度学习基于神经网络进行学习"),
            Document(id="4", content="自然语言处理让计算机理解人类语言"),
            Document(id="5", content="计算机视觉让机器能够看懂图像")
        ]
        
        api.add_documents("demo_retriever", documents)
        print(f"✓ 成功添加 {len(documents)} 个文档")
        
        # 3. 执行搜索
        print("\n3. 执行搜索...")
        queries = ["人工智能技术", "机器学习方法", "神经网络"]
        
        for query in queries:
            print(f"\n查询: '{query}'")
            results = api.search("demo_retriever", query, k=3)
            
            for i, doc in enumerate(results, 1):
                score = doc.metadata.get('score', 'N/A') if doc.metadata else 'N/A'
                print(f"  {i}. [ID:{doc.id}] {doc.content} (分数: {score})")
        
        # 4. 获取检索器信息
        print("\n4. 检索器信息:")
        info = api.get_retriever_info("demo_retriever")
        print(f"  类型: {info.get('type')}")
        print(f"  类名: {info.get('class')}")
        
        print("\n✓ 快速开始示例完成！")
        
    except Exception as e:
        print(f"✗ 示例执行失败: {e}")
        import traceback
        traceback.print_exc()

def alternative_creation():
    """替代创建方式示例"""
    print("\n=== 替代创建方式 ===")
    
    try:
        # 直接使用配置字典创建
        import tempfile
        import os
        temp_dir = tempfile.mkdtemp()
        bm25_index_path = os.path.join(temp_dir, "quick_bm25_index")

        config = {
            "type": "tantivy_bm25",
            "index_config": {
                "type": "bm25_indexer",
                "index_path": bm25_index_path,
                "language": "chinese"
            },
            "search_kwargs": {
                "k": 3,
                "with_score": True
            }
        }

        print(f"使用临时索引路径: {bm25_index_path}")
        
        api.create_retriever("bm25_demo", "tantivy_bm25", config)
        print("✓ BM25检索器创建成功")

        # 添加文档并搜索
        documents = [
            Document(id="1", content="文本检索是信息检索的重要技术"),
            Document(id="2", content="BM25算法在搜索引擎中广泛应用"),
            Document(id="3", content="关键词匹配是传统检索的基础")
        ]

        # 对于BM25，需要先初始化索引
        api.initialize_index("bm25_demo", documents)
        results = api.search("bm25_demo", "检索技术", k=2)
        
        print("搜索结果:")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.content}")
            
    except Exception as e:
        print(f"✗ 替代创建方式失败: {e}")

def cleanup():
    """清理示例"""
    print("\n=== 清理 ===")
    
    retrievers = api.list_retrievers()
    for name in retrievers:
        api.remove_retriever(name)
        print(f"✓ 移除检索器: {name}")

if __name__ == "__main__":
    quick_start()
    alternative_creation()
    cleanup()
    print("\n所有示例完成！")
