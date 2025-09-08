"""
Comprehensive retrieval testing code

Tests all retrieval functionality of the TantivyBM25Retriever
"""
import json
import os
import sys
import tempfile
import shutil
from typing import List

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.utils.data_model import Document
from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig, BM25IndexBuilder
from core.retrieval.tantivy_bm25 import TantivyBM25RetrieverConfig


def create_test_documents() -> List[Document]:
    """Create test documents for retrieval testing"""
    return [
        Document(
            id="tech_001",
            content="Python is a high-level programming language widely used for machine learning and data science applications.",
            metadata={"category": "technology", "language": "english", "difficulty": "intermediate", "author": "tech_writer"}
        ),
        Document(
            id="tech_002", 
            content="Deep learning neural networks can solve complex problems like image recognition and natural language processing.",
            metadata={"category": "technology", "language": "english", "difficulty": "advanced", "author": "ai_expert"}
        ),
        Document(
            id="science_001",
            content="Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways.",
            metadata={"category": "science", "language": "english", "difficulty": "advanced", "author": "physicist"}
        ),
        Document(
            id="chinese_001",
            content="机器学习是人工智能的重要分支，包括监督学习和无监督学习等多种方法。",
            metadata={"category": "technology", "language": "chinese", "difficulty": "intermediate", "author": "ai_researcher"}
        ),
        Document(
            id="chinese_002",
            content="深度学习使用神经网络来解决复杂的模式识别问题，在图像处理和自然语言处理领域有广泛应用。",
            metadata={"category": "technology", "language": "chinese", "difficulty": "advanced", "author": "ml_engineer"}
        ),
        Document(
            id="mixed_001",
            content="Python编程语言 combined with machine learning libraries like TensorFlow and PyTorch 为深度学习提供了强大的工具。",
            metadata={"category": "technology", "language": "mixed", "difficulty": "intermediate", "author": "developer"}
        ),
        Document(
            id="business_001",
            content="Artificial intelligence is transforming business operations across industries, from healthcare to finance.",
            metadata={"category": "business", "language": "english", "difficulty": "beginner", "author": "business_analyst"}
        ),
        Document(
            id="tutorial_001",
            content="Getting started with machine learning: understand the basics of supervised and unsupervised learning algorithms.",
            metadata={"category": "tutorial", "language": "english", "difficulty": "beginner", "author": "educator"}
        )
    ]



def test_comprehensive_retrieval():
    """综合检索功能测试 - 整合所有测试到一个函数中"""
    print("🚀 开始综合检索测试")
    print("=" * 60)
    
    all_passed = []
    all_failed = []
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            documents = create_test_documents()
            
            # === 测试 1: 基础检索功能 ===
            print("\n=== 测试 1: 基础检索功能 ===")
            try:
                index_path = os.path.join(temp_dir, "basic_retrieval_index")
                config = BM25IndexBuilderConfig(index_path=index_path)
                builder = config.build()
                builder.from_documents(documents)
                retriever = builder.as_retriever()
                
                # 基础搜索
                results = retriever.invoke("machine learning", k=3)
                if len(results) > 0:
                    print(f"✅ 基础搜索返回 {len(results)} 个结果")
                    all_passed.append("basic_search_returns_results")
                else:
                    print("❌ 基础搜索未返回结果")
                    all_failed.append("basic_search_returns_results")
                
                # 检查结果结构
                if results:
                    first_result = results[0]
                    if hasattr(first_result, 'id') and hasattr(first_result, 'content') and hasattr(first_result, 'metadata'):
                        print("✅ 结果结构正确 (id, content, metadata)")
                        all_passed.append("result_structure_correct")
                    else:
                        print("❌ 结果缺少必要字段")
                        all_failed.append("result_structure_correct")
                    
                    # 检查分数
                    if 'score' in first_result.metadata:
                        print(f"✅ 元数据中包含分数: {first_result.metadata['score']}")
                        all_passed.append("score_included")
                    else:
                        print("❌ 元数据中未找到分数")
                        all_failed.append("score_included")
                
                builder.close()
            except Exception as e:
                print(f"❌ 基础检索测试失败: {e}")
                all_failed.append("basic_retrieval_overall")
            
            # === 测试 2: 过滤检索功能 ===
            print("\n=== 测试 2: 过滤检索功能 ===")
            try:
                index_path = os.path.join(temp_dir, "filtered_retrieval_index")
                config = BM25IndexBuilderConfig(index_path=index_path)
                builder = config.build()
                builder.from_documents(documents)
                retriever = builder.as_retriever()
                
                # 单字段过滤
                tech_results = retriever.invoke("learning", filters={"category": "technology"}, k=5)
                all_tech = all(doc.metadata.get("category") == "technology" for doc in tech_results)
                if all_tech:
                    print(f"✅ 技术类别过滤返回 {len(tech_results)} 个正确结果")
                    all_passed.append("single_filter_correct")
                else:
                    print("❌ 部分过滤结果类别错误")
                    all_failed.append("single_filter_correct")
                
                # 多字段过滤
                advanced_tech_results = retriever.invoke(
                    "neural networks", 
                    filters={"category": "technology", "difficulty": "advanced"}, 
                    k=5
                )
                all_match = all(
                    doc.metadata.get("category") == "technology" and 
                    doc.metadata.get("difficulty") == "advanced" 
                    for doc in advanced_tech_results
                )
                if all_match:
                    print(f"✅ 多字段过滤返回 {len(advanced_tech_results)} 个正确结果")
                    all_passed.append("multiple_filters_correct")
                else:
                    print("❌ 部分结果不匹配所有过滤条件")
                    all_failed.append("multiple_filters_correct")
                
                builder.close()
            except Exception as e:
                print(f"❌ 过滤检索测试失败: {e}")
                all_failed.append("filtered_retrieval_overall")
            
            # === 测试 3: 短语查询功能 ===
            print("\n=== 测试 3: 短语查询功能 ===")
            try:
                index_path = os.path.join(temp_dir, "phrase_query_index")
                config = BM25IndexBuilderConfig(
                    index_path=index_path,
                    retrieval_use_phrase_query=True
                )
                builder = config.build()
                builder.from_documents(documents)
                retriever = builder.as_retriever()
                
                # 短语查询
                phrase_results = retriever.invoke("machine learning", k=5)
                if len(phrase_results) > 0:
                    print(f"✅ 短语查询返回 {len(phrase_results)} 个结果")
                    all_passed.append("phrase_query_returns_results")
                else:
                    print("❌ 短语查询未返回结果")
                    all_failed.append("phrase_query_returns_results")
                
                # 运行时覆盖短语查询设置
                normal_results = retriever.invoke("machine learning", k=5, use_phrase_query=False)
                phrase_override_results = retriever.invoke("machine learning", k=5, use_phrase_query=True)
                
                if len(normal_results) > 0 and len(phrase_override_results) > 0:
                    print("✅ 普通查询和短语查询覆盖都有效")
                    all_passed.append("phrase_query_override_works")
                else:
                    print("❌ 短语查询覆盖功能不正常")
                    all_failed.append("phrase_query_override_works")
                
                builder.close()
            except Exception as e:
                print(f"❌ 短语查询测试失败: {e}")
                all_failed.append("phrase_query_overall")
            
            # === 测试 4: 多语言检索功能 ===
            print("\n=== 测试 4: 多语言检索功能 ===")
            try:
                index_path = os.path.join(temp_dir, "multilingual_index")
                config = BM25IndexBuilderConfig(index_path=index_path)
                builder = config.build()
                builder.from_documents(documents)
                
                # 检查分词器选择
                tokenizer_info = builder.tokenizer_manager.get_tokenizer_info()
                print(f"✅ 选择的分词器: {tokenizer_info}")
                
                retriever = builder.as_retriever()
                
                # 英文查询
                english_results = retriever.invoke("machine learning", k=5)
                if len(english_results) > 0:
                    print(f"✅ 英文查询返回 {len(english_results)} 个结果")
                    all_passed.append("english_query_works")
                else:
                    print("❌ 英文查询失败")
                    all_failed.append("english_query_works")
                
                # 中文查询
                chinese_results = retriever.invoke("机器学习", k=5)
                if len(chinese_results) > 0:
                    print(f"✅ 中文查询返回 {len(chinese_results)} 个结果")
                    all_passed.append("chinese_query_works")
                else:
                    print("❌ 中文查询失败")
                    all_failed.append("chinese_query_works")
                
                # 混合语言查询
                mixed_results = retriever.invoke("Python 编程", k=5)
                if len(mixed_results) > 0:
                    print(f"✅ 混合语言查询返回 {len(mixed_results)} 个结果")
                    all_passed.append("mixed_language_query_works")
                else:
                    print("❌ 混合语言查询失败")
                    all_failed.append("mixed_language_query_works")
                
                builder.close()
            except Exception as e:
                print(f"❌ 多语言检索测试失败: {e}")
                all_failed.append("multilingual_retrieval_overall")
            
            # === 测试 5: 评分和排序功能 ===
            print("\n=== 测试 5: 评分和排序功能 ===")
            try:
                index_path = os.path.join(temp_dir, "scoring_index")
                config = BM25IndexBuilderConfig(
                    index_path=index_path,
                    retrieval_with_score=True
                )
                builder = config.build()
                builder.from_documents(documents)
                retriever = builder.as_retriever()
                
                # 包含分数的结果
                results_with_score = retriever.invoke("machine learning", k=5)
                
                if results_with_score and 'score' in results_with_score[0].metadata:
                    print("✅ 结果中包含分数")
                    all_passed.append("scores_included")
                    
                    # 检查分数是否按降序排列
                    scores = [doc.metadata['score'] for doc in results_with_score]
                    is_descending = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
                    
                    if is_descending:
                        print("✅ 结果按分数降序排列")
                        all_passed.append("results_ranked_by_score")
                    else:
                        print("❌ 结果未按分数正确排序")
                        all_failed.append("results_ranked_by_score")
                    
                    print(f"分数范围: {min(scores):.4f} 到 {max(scores):.4f}")
                else:
                    print("❌ 结果中未找到分数")
                    all_failed.append("scores_included")
                
                builder.close()
            except Exception as e:
                print(f"❌ 评分和排序测试失败: {e}")
                all_failed.append("scoring_ranking_overall")
            
            # === 测试 6: 检索器配置功能 ===
            print("\n=== 测试 6: 检索器配置功能 ===")
            try:
                index_path = os.path.join(temp_dir, "config_test_index")
                
                # 默认配置
                default_config = BM25IndexBuilderConfig(index_path=index_path)
                builder = default_config.build()
                builder.from_documents(documents)
                retriever = builder.as_retriever()
                
                default_results = retriever.invoke("machine learning")
                if len(default_results) <= 10:  # 默认 k=10
                    print("✅ 默认配置正常工作")
                    all_passed.append("default_config_works")
                else:
                    print("❌ 默认配置不正常")
                    all_failed.append("default_config_works")
                
                builder.close()
                
                # 自定义配置
                custom_config = BM25IndexBuilderConfig(
                    index_path=index_path + "_custom",
                    retrieval_k=3,
                    retrieval_use_phrase_query=True,
                    retrieval_with_score=False
                )
                builder_custom = custom_config.build()
                builder_custom.from_documents(documents)
                retriever_custom = builder_custom.as_retriever()
                
                custom_results = retriever_custom.invoke("machine learning")
                
                # 检查自定义 k 是否生效
                if len(custom_results) <= 3:
                    print("✅ 自定义 k 配置有效")
                    all_passed.append("custom_k_config_works")
                else:
                    print("❌ 自定义 k 配置无效")
                    all_failed.append("custom_k_config_works")
                
                builder_custom.close()
            except Exception as e:
                print(f"❌ 检索器配置测试失败: {e}")
                all_failed.append("retriever_config_overall")
            
            # === 测试 7: 错误处理功能 ===
            print("\n=== 测试 7: 错误处理功能 ===")
            try:
                index_path = os.path.join(temp_dir, "error_handling_index")
                config = BM25IndexBuilderConfig(index_path=index_path)
                builder = config.build()
                builder.from_documents(documents)
                retriever = builder.as_retriever()
                
                # 无效的 k 参数
                try:
                    invalid_k_results = retriever.invoke("test", k=0)
                    print("❌ 应该拒绝 k=0")
                    all_failed.append("invalid_k_handling")
                except ValueError as e:
                    if "must be greater than 0" in str(e):
                        print("✅ 正确处理无效的 k 参数")
                        all_passed.append("invalid_k_handling")
                    else:
                        print(f"❌ k=0 的错误消息不正确: {e}")
                        all_failed.append("invalid_k_handling")
                except Exception as e:
                    print(f"❌ k=0 的意外错误: {e}")
                    all_failed.append("invalid_k_handling")
                
                # 无效过滤字段
                try:
                    invalid_filter_results = retriever.invoke(
                        "test", 
                        filters={"nonexistent_field": "value"}
                    )
                    print("✅ 优雅地处理无效过滤字段")
                    all_passed.append("invalid_filter_handling")
                except Exception as e:
                    print(f"❌ 应该优雅地处理无效过滤器: {e}")
                    all_failed.append("invalid_filter_handling")
                
                # 特殊字符查询
                special_char_queries = [
                    "machine-learning",
                    "AI/ML",
                    "deep_learning",
                    "neural@networks",
                    "python3.8"
                ]
                
                special_char_success = 0
                for query in special_char_queries:
                    try:
                        results = retriever.invoke(query, k=3)
                        special_char_success += 1
                    except Exception:
                        pass
                
                if special_char_success == len(special_char_queries):
                    print("✅ 处理所有特殊字符查询")
                    all_passed.append("special_char_handling")
                elif special_char_success > 0:
                    print(f"⚠️ 处理 {special_char_success}/{len(special_char_queries)} 个特殊字符查询")
                    all_passed.append("partial_special_char_handling")
                else:
                    print("❌ 无法处理特殊字符查询")
                    all_failed.append("special_char_handling")
                
                builder.close()
            except Exception as e:
                print(f"❌ 错误处理测试失败: {e}")
                all_failed.append("error_handling_overall")
    
    except Exception as e:
        print(f"❌ 综合测试失败: {e}")
        all_failed.append("comprehensive_test_overall")
    
    # 测试总结
    print("\n" + "=" * 60)
    print("📊 综合检索测试总结")
    print("=" * 60)
    print(f"通过的测试总数: {len(all_passed)}")
    print(f"失败的测试总数: {len(all_failed)}")
    
    if all_failed:
        print("\n❌ 失败的测试:")
        for test in all_failed:
            print(f"  - {test}")
    
    if len(all_passed) > 0:
        success_rate = len(all_passed) / (len(all_passed) + len(all_failed)) * 100
        print(f"\n✅ 成功率: {success_rate:.1f}%")
    
    print("=" * 60)
    
    return len(all_failed) == 0



if __name__ == "__main__":
    print('开始综合检索测试')
    success = test_comprehensive_retrieval()
    if success:
        print("\n🎉 所有检索测试都通过了！")
        sys.exit(0)
    else:
        print("\n❌ 部分检索测试失败。请检查上面的输出。")
        sys.exit(1)
