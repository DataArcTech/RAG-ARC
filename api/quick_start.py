import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.retrieval_api import api
from encapsulation.data_model.schema import Chunk

def demo_dense_retriever():
    """演示Dense检索器"""
    print("=== Dense检索器演示 ===")

    try:
        # 1. 创建Dense检索器
        print("1. 创建Dense检索器...")
        api.create_from_config_file(
            "dense_retriever",
            "api/config_examples/unified_dense_config.json"
        )
        print("Dense检索器创建成功")

        # 2. 执行搜索
        print("\n2. 执行Dense检索...")
        queries = ["Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?", "Who is the figure associated with generative AI technology whose departure from OpenAI was considered shocking according to Fortune, and is also the subject of a prevailing theory suggesting a lack of full truthfulness with the board as reported by TechCrunch?"]

        for query in queries:
            print(f"\n查询: '{query}'")
            results = api.search("dense_retriever", query, k=3)

            for i, chunk in enumerate(results, 1):
                score = chunk.metadata.get('score', 'N/A') if chunk.metadata else 'N/A'
                print(f"  {i}. [ID:{chunk.id}] {chunk.content[:60]}... (分数: {score})")

        # 3. 获取检索器信息
        print("\n3. Dense检索器信息:")
        info = api.get_retriever_info("dense_retriever")
        print(f"  类型: {info.get('type')}")
        print(f"  类名: {info.get('class')}")

        print("\nDense检索器演示完成！")

    except Exception as e:
        print(f"Dense检索器演示失败: {e}")
        import traceback
        traceback.print_exc()


def demo_bm25_retriever():
    """演示BM25检索器"""
    print("\n=== BM25检索器演示 ===")

    try:
        # 1. 创建BM25检索器
        print("1. 创建BM25检索器...")
        api.create_from_config_file(
            "bm25_retriever",
            "api/config_examples/unified_bm25_config.json"
        )
        print("BM25检索器创建成功")


        # 3. 执行搜索
        print("\n3. 执行BM25检索...")
        queries = ["Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?", "Who is the figure associated with generative AI technology whose departure from OpenAI was considered shocking according to Fortune, and is also the subject of a prevailing theory suggesting a lack of full truthfulness with the board as reported by TechCrunch?"]

        for query in queries:
            print(f"\n查询: '{query}'")
            try:
                results = api.search("bm25_retriever", query, k=3)
                print(f"找到 {len(results)} 个结果:")

                if results:
                    for i, chunk in enumerate(results, 1):
                        score = chunk.metadata.get('score', 'N/A') if chunk.metadata else 'N/A'
                        content_preview = chunk.content[:80].replace('\n', ' ')
                        print(f"  {i}. [ID:{chunk.id}] {content_preview}... (分数: {score})")
                else:
                    print("  没有找到相关结果")
            except Exception as e:
                print(f"  搜索出错: {e}")

        # 4. 获取检索器信息
        print("\n4. BM25检索器信息:")
        info = api.get_retriever_info("bm25_retriever")
        print(f"  类型: {info.get('type')}")
        print(f"  类名: {info.get('class')}")

        print("\nBM25检索器演示完成！")

    except Exception as e:
        print(f"BM25检索器演示失败: {e}")
        import traceback
        traceback.print_exc()


def demo_multipath_retriever():
    """演示MultiPath检索器"""
    print("\n=== MultiPath检索器演示 ===")

    try:
        # 1. 创建MultiPath检索器
        print("1. 创建MultiPath检索器...")
        api.create_from_config_file(
            "multipath_retriever",
            "api/config_examples/unified_multipath_config.json"
        )
        print("MultiPath检索器创建成功")

        # 2. 执行搜索
        print("\n2. 执行MultiPath检索...")
        queries = ["Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?", "Who is the figure associated with generative AI technology whose departure from OpenAI was considered shocking according to Fortune, and is also the subject of a prevailing theory suggesting a lack of full truthfulness with the board as reported by TechCrunch?"]

        for query in queries:
            print(f"\n查询: '{query}'")
            results = api.search("multipath_retriever", query, k=3)

            for i, chunk in enumerate(results, 1):
                score = chunk.metadata.get('score', 'N/A') if chunk.metadata else 'N/A'
                print(f"  {i}. [ID:{chunk.id}] {chunk.content[:60]}... (分数: {score})")

        # 3. 获取检索器信息
        print("\n3. MultiPath检索器信息:")
        info = api.get_retriever_info("multipath_retriever")
        print(f"  类型: {info.get('type')}")
        print(f"  类名: {info.get('class')}")

        # 4. 获取多路径详细信息
        retriever = api.retrievers.get("multipath_retriever")
        if retriever and hasattr(retriever, 'get_multipath_info'):
            multipath_info = retriever.get_multipath_info()
            print(f"  子检索器数量: {multipath_info.get('retriever_count', 0)}")
            print(f"  子检索器类型: {multipath_info.get('retriever_types', [])}")
            print(f"  融合方法: {multipath_info.get('fusion_method', 'N/A')}")

        print("\nMultiPath检索器演示完成！")

    except Exception as e:
        print(f"MultiPath检索器演示失败: {e}")
        import traceback
        traceback.print_exc()


def quick_start():
    """快速开始示例 - 演示所有检索器类型"""
    print("=== 统一检索API快速开始 ===")
    print("本示例将演示三种检索器类型：Dense、BM25和MultiPath")

    # 演示Dense检索器
    demo_dense_retriever()

    # 演示BM25检索器
    demo_bm25_retriever()

    # 演示MultiPath检索器
    demo_multipath_retriever()

    print("\n所有检索器演示完成！")


if __name__ == "__main__":
    quick_start()
    print("\n所有示例完成！")
