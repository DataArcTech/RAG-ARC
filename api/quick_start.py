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


if __name__ == "__main__":
    quick_start()
    # alternative_creation()
    # cleanup()
    print("\n所有示例完成！")
