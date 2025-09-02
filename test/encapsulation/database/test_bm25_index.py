import os
import shutil
from typing import List
import sys
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from core.utils.data_model import Document
from encapsulation.database.bm25_indexer import BM25IndexBuilder


# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def run_test():
    """完整测试所有功能"""
    index_dir = "./test_bm25_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)  # 清理旧索引目录
    
    logger.info("=== 开始完整功能测试 ===")
    
    # ========== 1. 中文文档（验证 jieba） ==========
    chinese_docs = [
        Document(id="1", content="苹果公司是一家科技企业，位于中国", metadata={"source": "12", "tag": "company"}),
        Document(id="2", content="苹果不仅是水果，还是手机品牌", metadata={"source": "23", "tag": "fruit"}),
        Document(id="3", content="微软公司的主要产品是Windows和Office", metadata={"source": "wiki", "tag": "company"}),
        Document(id="4", content="特斯拉公司生产电动汽车和能源产品", metadata={"source": "wiki", "tag": "company"}),
        Document(id="5", content="李雷和韩梅梅是中学生的名字", metadata={"source": "wiki", "tag": "person"}),
        Document(id="6", content="北京是中国的首都，也是政治文化中心", metadata={"source": "wiki", "tag": "location"}),
        Document(id="7", content="苹果公司推出了新的iPhone产品，销量不错", metadata={"source": "wiki", "tag": "company"}),
        Document(id="8", content="微软推出了新的Surface笔记本电脑", metadata={"source": "wiki", "tag": "company"}),
    ]

    logger.info("=== 1. 构建中文索引 (jieba) ===")
    builder = BM25IndexBuilder.from_documents(chinese_docs, index_path=index_dir)
    logger.info(f"索引统计: {builder.get_index_stats()}")
    print(f"索引统计: {builder.get_index_stats()}")

    # 获取分词器信息
    tokenizer_stats = builder.get_tokenizer_stats()
    logger.info(f"分词器信息: {tokenizer_stats}")
    print(f"分词器信息: {tokenizer_stats}")

    # 检索验证中文
    retriever = builder.as_retriever(top_k=4)
    results = retriever.invoke("苹果公司推出了新的iPhone产品，销量不错")
    print("\n=== 1.1 中文检索结果 ===")
    for r in results:
        print(f"ID={r.id}, 内容={r.content}, 元数据={r.metadata}")

    print("\n=== 1.2 中文检索结果 (元数据过滤 - tag='fruit') ===")
    results = retriever.invoke("苹果公司推出了新的iPhone产品，销量不错", filters={"tag": "fruit"})
    for r in results:
        print(f"ID={r.id}, 内容={r.content}, 元数据={r.metadata}")

    print("\n=== 1.3 中文检索结果 (元数据过滤 - tag='company' and source='12') ===")
    results = retriever.invoke("苹果公司推出了新的iPhone产品，销量不错", filters={"tag": "company", "source": "12"})
    for r in results:
        print(f"ID={r.id}, 内容={r.content}, 元数据={r.metadata}")

    # 测试获取单个文档
    print("\n=== 1.4 获取单个文档 (ID='1') ===")
    doc = builder.get_document_by_id("1")
    if doc:
        print(f"获取到文档: ID={doc.id}, 内容={doc.content}, 元数据={doc.metadata}")
    else:
        print("未找到文档")

    # ========== 2. 增量添加英文文档 ==========
    english_docs = [
        Document(id="9", content="Apple is both a fruit and a technology company", metadata={"source": "wiki", "tag": "english"}),
        Document(id="10", content="Google develops the Android operating system", metadata={"source": "wiki", "tag": "english"}),
        Document(id="11", content="Microsoft produces Windows and Office software", metadata={"source": "wiki", "tag": "english"}),
        Document(id="12", content="Tesla manufactures electric cars and energy products", metadata={"source": "wiki", "tag": "english"}),
    ]


    logger.info("=== 2. 增量添加英文文档 (whitespace) ===")
    added_ids = builder.add_documents(english_docs)
    logger.info(f"新增文档ID: {added_ids}")
    logger.info(f"更新后的索引统计: {builder.get_index_stats()}")
    print(f"更新后的索引统计: {builder.get_index_stats()}")

    # 检索验证英文
    retriever = builder.as_retriever(top_k=3)
    results = retriever.invoke("Android system")
    print("\n=== 2.1 英文检索结果 ===")
    for r in results:
        print(f"ID={r.id}, 内容={r.content}")

    # ========== 3. 覆盖更新 ==========
    logger.info("=== 3. 覆盖更新文档 ===")
    updated_docs = [
        Document(id="2", content="苹果是著名水果，也是一家手机公司", metadata={"source": "wiki", "tag": "updated"}),
        Document(id="5", content="李雷和韩梅梅是中国学生", metadata={"source": "wiki", "tag": "updated"}),
        Document(id="10", content="Google develops Android and Chrome products", metadata={"source": "wiki", "tag": "updated"}),
    ]

    updated_ids = builder.update_documents(updated_docs)
    logger.info(f"更新文档ID: {updated_ids}")
    logger.info(f"更新后的索引统计: {builder.get_index_stats()}")

    # 验证更新结果
    print("\n=== 3.1 验证更新结果 ===")
    doc2 = builder.get_document_by_id("2")
    if doc2:
        print(f"文档2更新后: {doc2.content}, 元数据: {doc2.metadata}")
    else:
        print("文档2未找到")

    doc10 = builder.get_document_by_id("10")
    if doc10:
        print(f"文档10更新后: {doc10.content}, 元数据: {doc10.metadata}")
    else:
        print("文档10未找到")

    # ========== 4. 删除 ==========
    logger.info("=== 4. 删除文档 ===")
    deleted_count = builder.delete_documents(["1", "4", "4", "7"])  # 包含重复ID
    logger.info(f"删除数量: {deleted_count}")
    logger.info(f"删除后的索引统计: {builder.get_index_stats()}")

    # 验证删除结果
    print("\n=== 4.1 验证删除结果 ===")
    deleted_doc = builder.get_document_by_id("1")
    if deleted_doc is None:
        print("文档1已成功删除")
    else:
        print("文档1删除失败")

    # ========== 5. 混合检索 ==========
    retriever = builder.as_retriever(top_k=5)
    results = retriever.invoke("苹果 公司 Android")
    print("\n=== 5.1 混合检索结果 (中英文混合查询) ===")
    for r in results:
        print(f"ID={r.id}, 内容={r.content}")
    
    # ========== 6. 调试：查看所有文档 ==========
    print("\n=== 6.1 调试：查看所有文档 ===")
    # 尝试通用查询
    general_results = retriever.invoke("公司")
    print(f"'公司'查询返回文档数: {len(general_results)}")
    for r in general_results:
        print(f"ID={r.id}, 内容={r.content}")
    
    # 尝试空查询（应该报错）
    print(f"\n=== 6.2 测试空查询（应该报错） ===")
    try:
        empty_results = retriever.invoke("")
        print(f"空查询返回文档数: {len(empty_results)}")
    except Exception as e:
        print(f"空查询异常: {e}")

    # ========== 7. 真实数据测试 ==========
    try:
        print("\n=== 7. 真实数据测试 ===")
        import json
        docs = []
        test_file_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/test/tcl_gb_chunk.json"
        
        if os.path.exists(test_file_path):
            with open(test_file_path, "r") as f:
                data = json.load(f)
                for i, item in enumerate(data):  # 只取前10条进行测试
                    metadata = {"source": item["metadata"]["file_name"]}
                    docs.append(Document(id=str(i), content=item["content"], metadata=metadata))

            if docs:
                # 创建新的索引器进行真实数据测试
                real_data_index_dir = "./test_real_data_index"
                if os.path.exists(real_data_index_dir):
                    shutil.rmtree(real_data_index_dir)
                
                real_builder = BM25IndexBuilder(index_path=real_data_index_dir)
                added_ids = real_builder.add_documents(docs)
                print(f"真实数据索引统计: {real_builder.get_index_stats()}")
                
                real_retriever = real_builder.as_retriever(top_k=3)
                results = real_retriever.invoke("冷凝器片距的选择依据是什么？")
                print("\n=== 7.1 真实数据检索结果 ===")
                for r in results:
                    print(f"ID={r.id}, 内容={r.content[:100]}...")  # 只显示前100个字符
                
                # 测试带过滤器的检索
                if len(data) > 0 and "metadata" in data[0] and "file_name" in data[0]["metadata"]:
                    sample_source = data[0]["metadata"]["file_name"]
                    results_filtered = real_retriever.invoke("冷凝器", filters={"source": sample_source})
                    print(f"\n=== 7.2 真实数据带过滤器检索结果 (source='{sample_source}') ===")
                    for r in results_filtered:
                        print(f"ID={r.id}, 内容={r.content[:100]}...")
                
                # 关闭真实数据索引器
                real_builder.close()
                if os.path.exists(real_data_index_dir):
                    shutil.rmtree(real_data_index_dir)
            else:
                print("真实数据文件为空或格式不正确")
        else:
            print(f"真实数据文件不存在: {test_file_path}")
            
    except Exception as e:
        print(f"真实数据测试异常: {e}")

    # ========== 8. 上下文管理器测试 ==========
    print("\n=== 8. 上下文管理器测试 ===")
    context_index_dir = "./test_context_index"
    if os.path.exists(context_index_dir):
        shutil.rmtree(context_index_dir)
    
    try:
        with BM25IndexBuilder(index_path=context_index_dir) as context_builder:
            # 添加一些测试文档
            context_docs = [
                Document(id="ctx1", content="上下文管理器测试文档1", metadata={"test": "context"}),
                Document(id="ctx2", content="上下文管理器测试文档2", metadata={"test": "context"}),
            ]
            context_builder.add_documents(context_docs)
            print(f"上下文管理器内索引统计: {context_builder.get_index_stats()}")
            
            # 测试检索
            context_retriever = context_builder.as_retriever(top_k=2)
            context_results = context_retriever.invoke("测试")
            print("上下文管理器检索结果:")
            for r in context_results:
                print(f"ID={r.id}, 内容={r.content}")
            
            # 异常测试
            raise ValueError("模拟异常测试上下文管理器的异常处理")
    except ValueError as e:
        print(f"捕获到预期异常: {e}")
    
    # 验证上下文管理器是否正确关闭
    print("上下文管理器测试完成，验证资源是否释放")
    

    # ========== 9. 自定义预处理函数测试 ==========
    print("\n=== 9. 自定义预处理函数测试 ===")
    custom_index_dir = "./test_custom_preprocess_index"
    if os.path.exists(custom_index_dir):
        shutil.rmtree(custom_index_dir)
    
    def custom_tokenize(text: str) -> List[str]:
        """自定义分词函数"""
        if not text:
            return []
        # 简单按字符分割
        return list(text.replace(" ", ""))
    
    custom_builder = BM25IndexBuilder(
        index_path=custom_index_dir,
        preprocess_func=custom_tokenize
    )
    
    custom_docs = [
        Document(id="c1", content="自定义  分词 测试1", metadata={"type": "custom"}),
        Document(id="c2", content="自定义分词测 试2", metadata={"type": "custom"}),
    ]
    
    custom_builder.add_documents(custom_docs)
    print(f"自定义预处理函数索引统计: {custom_builder.get_index_stats()}")
    print(f"分词器信息: {custom_builder.get_tokenizer_stats()}")
    
    # 测试检索
    custom_retriever = custom_builder.as_retriever(top_k=2)
    custom_results = custom_retriever.invoke("测试")
    print("自定义预处理函数检索结果:")
    for r in custom_results:
        print(f"ID={r.id}, 内容={r.content}")
    
    custom_builder.close()
    if os.path.exists(custom_index_dir):
        shutil.rmtree(custom_index_dir)

    # 关闭主 builder
    builder.close()
    
    # 清理测试目录
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    
    logger.info("=== 所有测试完成 ===")


if __name__ == "__main__":
    run_test()

