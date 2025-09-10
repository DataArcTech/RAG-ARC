
import json
import os
import sys
import tempfile
import shutil
import random
from typing import List

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.utils.data_model import Document
from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig, BM25IndexBuilder

from core.retrieval.multipath import MultiPathRetrieverConfig
from encapsulation.database.vector_db.faiss import FaissVectorDBConfig
from core.file_management.embeddings.base import Embeddings
from encapsulation.llm.base import LLMBase
from encapsulation.llm.huggingface import HuggingFaceEmbedConfig
from framework.register import Register


class MockEmbeddings(LLMBase):
    """Mock embeddings for testing"""
    
    def __init__(self, dimension=128):
        super().__init__()
        self.dimension = dimension
        # Use fixed seed for reproducible results
        self.random = random.Random(42)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Mock embedding documents with consistent results"""
        embeddings = []
        for text in texts:
            # Create deterministic embeddings based on text hash
            text_hash = hash(text)
            self.random.seed(text_hash)
            embedding = [self.random.random() for _ in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Mock embedding query with consistent results"""
        text_hash = hash(text)
        self.random.seed(text_hash)
        return [self.random.random() for _ in range(self.dimension)]
    
    # 实现LLMBase的抽象方法
    def _chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """Mock chat implementation - not used for embeddings"""
        return "Mock chat response"
    
    def _stream_chat(self, messages, max_tokens=None, temperature=None, **kwargs):
        """Mock streaming chat implementation - not used for embeddings"""
        yield "Mock streaming response"
    
    def _embed(self, texts):
        """Mock embed implementation"""
        if isinstance(texts, str):
            return self.embed_query(texts)
        else:
            return self.embed_documents(texts)
    
    def _rerank(self, query, documents, top_k=None):
        """Mock rerank implementation - not used for embeddings"""
        # Return mock reranking scores
        scores = [(i, random.random()) for i in range(len(documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            scores = scores[:top_k]
        return scores


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
                    search_kwargs={"use_phrase_query": True}
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
                    index_path=index_path
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
                    search_kwargs={"use_phrase_query": True}
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


def create_real_embedding_model():
    """创建真实的HuggingFace嵌入模型"""
    try:
        # 使用一个轻量级的嵌入模型进行测试
        config = HuggingFaceEmbedConfig(
            model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",  # 轻量级模型，适合测试
            device="cuda:0",
            task_types="embedding"  # 修复：task_types应该是字符串，不是列表
        )
        return config.build()
    except Exception as e:
        print(f"⚠️ 无法创建真实嵌入模型: {e}")
        print("📝 回退到Mock嵌入模型")
        return MockEmbeddings(dimension=384)  # all-MiniLM-L6-v2 的维度是384


def register_embedding_model(embedding_model, model_name="test_embedding"):
    """注册嵌入模型到注册表"""
    register = Register()
    
    # 创建临时配置文件
    import tempfile
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # 创建一个简单的配置，实际不会使用，因为我们直接传递实例
        config_data = {"model_name": "test", "device": "cpu", "task_types": ["embedding"]}
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        # 直接注册嵌入模型实例
        register.registrations[model_name] = embedding_model
        return model_name
    finally:
        # 清理临时文件
        import os
        try:
            os.unlink(config_path)
        except:
            pass


def test_dense_retrieval():
    """Dense检索功能测试"""
    
    print("🚀 开始Dense检索测试")
    print("=" * 60)
    
    all_passed = []
    all_failed = []
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            documents = create_test_documents()
            
            # === 测试 1: Dense检索器基础功能 ===
            print("\n=== 测试 1: Dense检索器基础功能 ===")
            try:
                # 创建真实的HuggingFace嵌入模型
                embedding_model = create_real_embedding_model()
                print(f"📊 使用嵌入模型: {embedding_model.__class__.__name__}")
                
                # 创建Faiss向量数据库配置
                vector_db_config = FaissVectorDBConfig(
                    embedding_config={
                        "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                        "task_types": "embedding",
                        "device": "cuda:0"
                    },
                    index_path=os.path.join(temp_dir, "dense_index"),
                    metric="cosine",
                    normalize_L2=True
                )
                
                # 构建向量数据库
                vector_db = vector_db_config.build()
                vector_db.from_documents(documents)
                
                # 创建Dense检索器
                retriever = vector_db.as_retriever()
                
                # 基础搜索
                results = retriever.invoke("machine learning", k=3)
                if len(results) > 0:
                    print(f"✅ Dense基础搜索返回 {len(results)} 个结果")
                    all_passed.append("dense_basic_search_returns_results")
                else:
                    print("❌ Dense基础搜索未返回结果")
                    all_failed.append("dense_basic_search_returns_results")
                
                # 检查结果结构
                if results:
                    first_result = results[0]
                    if hasattr(first_result, 'id') and hasattr(first_result, 'content') and hasattr(first_result, 'metadata'):
                        print("✅ Dense结果结构正确 (id, content, metadata)")
                        all_passed.append("dense_result_structure_correct")
                    else:
                        print("❌ Dense结果缺少必要字段")
                        all_failed.append("dense_result_structure_correct")
                    
                    # 检查分数
                    if 'score' in first_result.metadata:
                        print(f"✅ Dense元数据中包含分数: {first_result.metadata['score']}")
                        all_passed.append("dense_score_included")
                    else:
                        print("❌ Dense元数据中未找到分数")
                        all_failed.append("dense_score_included")
                
            except Exception as e:
                print(f"❌ Dense基础检索测试失败: {e}")
                all_failed.append("dense_basic_retrieval_overall")
            
            # === 测试 2: Dense检索器搜索类型 ===
            print("\n=== 测试 2: Dense检索器搜索类型 ===")
            try:
                vector_db_config = FaissVectorDBConfig(
                    embedding_config={
                        "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                        "task_types": "embedding",
                        "device": "cuda:0"
                    },
                    index_path=os.path.join(temp_dir, "dense_search_types"),
                    metric="cosine",
                    normalize_L2=True
                )
                vector_db = vector_db_config.build()
                vector_db.from_documents(documents)
                retriever = vector_db.as_retriever()
                
                # 相似性搜索
                similarity_results = retriever.invoke("machine learning", k=3, search_type="similarity")
                if len(similarity_results) > 0:
                    print(f"✅ 相似性搜索返回 {len(similarity_results)} 个结果")
                    all_passed.append("dense_similarity_search_works")
                else:
                    print("❌ 相似性搜索失败")
                    all_failed.append("dense_similarity_search_works")
                
                # MMR搜索
                mmr_results = retriever.invoke("machine learning", k=3, search_type="mmr", lambda_mult=0.5)
                if len(mmr_results) > 0:
                    print(f"✅ MMR搜索返回 {len(mmr_results)} 个结果")
                    all_passed.append("dense_mmr_search_works")
                else:
                    print("❌ MMR搜索失败")
                    all_failed.append("dense_mmr_search_works")
                
                # 分数阈值搜索
                threshold_results = retriever.invoke(
                    "machine learning", k=5, 
                    search_type="similarity_score_threshold", 
                    score_threshold=0.1
                )
                print(f"✅ 分数阈值搜索返回 {len(threshold_results)} 个结果")
                all_passed.append("dense_threshold_search_works")
                
            except Exception as e:
                print(f"❌ Dense搜索类型测试失败: {e}")
                all_failed.append("dense_search_types_overall")
            
            # === 测试 3: Dense检索器不同度量 ===
            print("\n=== 测试 3: Dense检索器不同度量 ===")
            try:
                # 测试不同度量
                metrics = ["cosine", "l2", "ip"]
                for metric in metrics:
                    try:
                        vector_db_config = FaissVectorDBConfig(
                            embedding_config={
                                "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                                "task_types": "embedding",
                                "device": "cuda:0"
                            },
                            index_path=os.path.join(temp_dir, f"dense_metric_{metric}"),
                            metric=metric,
                            normalize_L2=(metric == "cosine")
                        )
                        vector_db = vector_db_config.build()
                        vector_db.from_documents(documents)
                        retriever = vector_db.as_retriever()
                        
                        results = retriever.invoke("machine learning", k=3)
                        if len(results) > 0:
                            print(f"✅ {metric} 度量搜索返回 {len(results)} 个结果")
                            all_passed.append(f"dense_{metric}_metric_works")
                        else:
                            print(f"❌ {metric} 度量搜索失败")
                            all_failed.append(f"dense_{metric}_metric_works")
                    except Exception as e:
                        print(f"❌ {metric} 度量测试失败: {e}")
                        all_failed.append(f"dense_{metric}_metric_works")
                        
            except Exception as e:
                print(f"❌ Dense度量测试失败: {e}")
                all_failed.append("dense_metrics_overall")
            
            # === 测试 4: Dense检索器错误处理 ===
            print("\n=== 测试 4: Dense检索器错误处理 ===")
            try:
                vector_db_config = FaissVectorDBConfig(
                    embedding_config={
                        "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                        "task_types": "embedding",
                        "device": "cuda:0"
                    },
                    index_path=os.path.join(temp_dir, "dense_error_handling"),
                    metric="cosine"
                )
                vector_db = vector_db_config.build()
                vector_db.from_documents(documents)
                retriever = vector_db.as_retriever()
                
                # 无效的 k 参数
                try:
                    invalid_k_results = retriever.invoke("test", k=0)
                    print("❌ 应该拒绝 k=0")
                    all_failed.append("dense_invalid_k_handling")
                except ValueError as e:
                    if "ust be greater than 0" in str(e):
                        print("✅ 正确处理无效的 k 参数")
                        all_passed.append("dense_invalid_k_handling")
                    else:
                        print(f"❌ k=0 的错误消息不正确: {e}")
                        all_failed.append("dense_invalid_k_handling")
                
                # 无效搜索类型
                try:
                    invalid_search_results = retriever.invoke("test", search_type="invalid_type")
                    print("❌ 应该拒绝无效搜索类型")
                    all_failed.append("dense_invalid_search_type_handling")
                except ValueError as e:
                    if "is not allowed" in str(e):
                        print("✅ 正确处理无效搜索类型")
                        all_passed.append("dense_invalid_search_type_handling")
                    else:
                        print(f"❌ 无效搜索类型的错误消息不正确: {e}")
                        all_failed.append("dense_invalid_search_type_handling")
                
                # 空查询
                empty_results = retriever.invoke("", k=3)
                if len(empty_results) == 0:
                    print("✅ 正确处理空查询")
                    all_passed.append("dense_empty_query_handling")
                else:
                    print("❌ 空查询应该返回空结果")
                    all_failed.append("dense_empty_query_handling")
                
            except Exception as e:
                print(f"❌ Dense错误处理测试失败: {e}")
                all_failed.append("dense_error_handling_overall")
    
    except Exception as e:
        print(f"❌ Dense综合测试失败: {e}")
        all_failed.append("dense_comprehensive_test_overall")
    
    # 测试总结
    print("\n" + "=" * 60)
    print("📊 Dense检索测试总结")
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


def test_multipath_retrieval():
    """MultiPath检索功能测试 - 参考test_multipath_simple.py的方式"""
    
    print("🚀 开始MultiPath检索测试")
    print("=" * 60)
    
    all_passed = []
    all_failed = []
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            documents = create_test_documents()
            
            # === 测试 1: MultiPath检索器基础功能 ===
            print("\n=== 测试 1: MultiPath检索器基础功能 ===")
            try:
                # 创建测试配置文件
                config_data = {
                    "documents": [
                        {
                            "id": doc.id,
                            "content": doc.content,
                            "metadata": doc.metadata
                        }
                        for doc in documents
                    ],
                    "query": "machine learning",
                    "multipath_config": {
                        "type": "multipath",
                        "k": 5,
                        "with_score": True,
                        "top_k_per_retriever": 10,
                        "search_kwargs": {},
                        "indexers": [
                            {
                                "type": "bm25_indexer",
                                "index_path": os.path.join(temp_dir, "multipath_bm25_index"),
                                "bm25_k1": 1.2,
                                "bm25_b": 0.75
                            },
                            {
                                "type": "faiss",
                                "index_path": os.path.join(temp_dir, "multipath_faiss_index"),
                                "metric": "cosine",
                                "normalize_L2": True,
                                "embedding_config": {
                                    "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                                    "task_types": "embedding",
                                    "device": "cuda:0"
                                }
                            }
                        ]
                    }
                }
                
                # 使用model_validate直接从JSON数据创建MultiPathRetrieverConfig对象
                multipath_config = MultiPathRetrieverConfig.model_validate(config_data.get("multipath_config", {}))
                
                # 从multipath_config中获取faiss_config和bm25_config
                faiss_config = None
                bm25_config = None
                for indexer in multipath_config.indexers:
                    if indexer.type == "faiss":
                        faiss_config = indexer
                    elif indexer.type == "bm25_indexer":
                        bm25_config = indexer
                
                # 确保索引目录存在
                faiss_dir = faiss_config.index_path
                bm25_dir = bm25_config.index_path
                os.makedirs(faiss_dir, exist_ok=True)
                os.makedirs(bm25_dir, exist_ok=True)
                
                # 构建并保存 FAISS 向量索引
                faiss_db = faiss_config.build()
                faiss_db.from_documents(documents)
                
                # 构建并保存 BM25 索引
                bm25_builder = bm25_config.build()
                bm25_builder.from_documents(documents)
                
                # 初始化多路检索器
                mp = multipath_config.build()
                
                # 执行检索
                query = config_data["query"]
                results = mp.invoke(query)
                
                if len(results) > 0:
                    print(f"✅ MultiPath基础搜索返回 {len(results)} 个结果")
                    all_passed.append("multipath_basic_search_returns_results")
                else:
                    print("❌ MultiPath基础搜索未返回结果")
                    all_failed.append("multipath_basic_search_returns_results")
                
                # 检查结果结构
                if results:
                    first_result = results[0]
                    if hasattr(first_result, 'id') and hasattr(first_result, 'content') and hasattr(first_result, 'metadata'):
                        print("✅ MultiPath结果结构正确 (id, content, metadata)")
                        all_passed.append("multipath_result_structure_correct")
                    else:
                        print("❌ MultiPath结果缺少必要字段")
                        all_failed.append("multipath_result_structure_correct")
                    
                    # 检查融合分数
                    if 'score' in first_result.metadata:
                        print(f"✅ MultiPath元数据中包含融合分数: {first_result.metadata['score']}")
                        all_passed.append("multipath_fusion_score_included")
                    else:
                        print("❌ MultiPath元数据中未找到融合分数")
                        all_failed.append("multipath_fusion_score_included")
                
                # 清理资源
                bm25_builder.close()
                
            except Exception as e:
                print(f"❌ MultiPath基础检索测试失败: {e}")
                all_failed.append("multipath_basic_retrieval_overall")
            
            # === 测试 2: MultiPath检索器参数配置 ===
            print("\n=== 测试 2: MultiPath检索器参数配置 ===")
            try:
                # 测试不同的top_k_per_retriever配置
                for top_k_per_retriever in [5, 10, 20]:
                    try:
                        config_data = {
                            "documents": [
                                {
                                    "id": doc.id,
                                    "content": doc.content,
                                    "metadata": doc.metadata
                                }
                                for doc in documents
                            ],
                            "query": "machine learning",
                            "multipath_config": {
                                "type": "multipath",
                                "k": 3,
                                "with_score": True,
                                "top_k_per_retriever": top_k_per_retriever,
                                "search_kwargs": {},
                                "indexers": [
                                    {
                                        "type": "bm25_indexer",
                                        "index_path": os.path.join(temp_dir, f"multipath_config_bm25_{top_k_per_retriever}"),
                                        "bm25_k1": 1.2,
                                        "bm25_b": 0.75
                                    },
                                    {
                                        "type": "faiss",
                                        "index_path": os.path.join(temp_dir, f"multipath_config_faiss_{top_k_per_retriever}"),
                                        "metric": "cosine",
                                        "normalize_L2": True,
                                        "embedding_config": {
                                            "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                                            "task_types": "embedding",
                                            "device": "cuda:0"
                                        }
                                    }
                                ]
                            }
                        }
                        
                        # 使用model_validate直接从JSON数据创建MultiPathRetrieverConfig对象
                        multipath_config = MultiPathRetrieverConfig.model_validate(config_data.get("multipath_config", {}))
                        
                        # 从multipath_config中获取faiss_config和bm25_config
                        faiss_config = None
                        bm25_config = None
                        for indexer in multipath_config.indexers:
                            if indexer.type == "faiss":
                                faiss_config = indexer
                            elif indexer.type == "bm25_indexer":
                                bm25_config = indexer
                        
                        # 确保索引目录存在
                        faiss_dir = faiss_config.index_path
                        bm25_dir = bm25_config.index_path
                        os.makedirs(faiss_dir, exist_ok=True)
                        os.makedirs(bm25_dir, exist_ok=True)
                        
                        # 构建并保存 FAISS 向量索引
                        faiss_db = faiss_config.build()
                        faiss_db.from_documents(documents)
                        
                        # 构建并保存 BM25 索引
                        bm25_builder = bm25_config.build()
                        bm25_builder.from_documents(documents)
                        
                        # 初始化多路检索器
                        mp = multipath_config.build()
                        
                        # 执行检索
                        query = config_data["query"]
                        results = mp.invoke(query)
                        
                        if len(results) > 0:
                            print(f"✅ top_k_per_retriever={top_k_per_retriever} 返回 {len(results)} 个结果")
                            all_passed.append(f"multipath_top_k_{top_k_per_retriever}_works")
                        else:
                            print(f"❌ top_k_per_retriever={top_k_per_retriever} 失败")
                            all_failed.append(f"multipath_top_k_{top_k_per_retriever}_works")
                        
                        # 清理资源
                        bm25_builder.close()
                    except Exception as e:
                        print(f"❌ top_k_per_retriever={top_k_per_retriever} 配置测试失败: {e}")
                        all_failed.append(f"multipath_top_k_{top_k_per_retriever}_works")
                
            except Exception as e:
                print(f"❌ MultiPath配置测试失败: {e}")
                all_failed.append("multipath_config_overall")
            
            # === 测试 3: MultiPath检索器错误处理 ===
            print("\n=== 测试 3: MultiPath检索器错误处理 ===")
            try:
                # 测试空indexers的错误处理
                try:
                    config_data = {
                        "multipath_config": {
                            "type": "multipath",
                            "k": 3,
                            "with_score": True,
                            "top_k_per_retriever": 10,
                            "search_kwargs": {},
                            "indexers": []  # 空indexers
                        }
                    }
                    
                    # 这应该会抛出验证错误
                    multipath_config = MultiPathRetrieverConfig.model_validate(config_data.get("multipath_config", {}))
                    print("❌ 应该拒绝空indexers")
                    all_failed.append("multipath_empty_indexers_handling")
                except Exception as e:
                    if "At least one indexer config is required" in str(e):
                        print("✅ 正确处理空indexers错误")
                        all_passed.append("multipath_empty_indexers_handling")
                    else:
                        print(f"❌ 空indexers的错误消息不正确: {e}")
                        all_failed.append("multipath_empty_indexers_handling")
                
                # 测试无效type的错误处理
                try:
                    config_data = {
                        "multipath_config": {
                            "type": "multipath",
                            "k": 3,
                            "with_score": True,
                            "top_k_per_retriever": 10,
                            "search_kwargs": {},
                            "indexers": [
                                {
                                    "type": "invalid_type",  # 无效type
                                    "index_path": os.path.join(temp_dir, "multipath_invalid_type"),
                                }
                            ]
                        }
                    }
                    
                    # 这应该会抛出验证错误
                    multipath_config = MultiPathRetrieverConfig.model_validate(config_data.get("multipath_config", {}))
                    print("❌ 应该拒绝无效type")
                    all_failed.append("multipath_invalid_type_handling")
                except Exception as e:
                    print("✅ 正确处理无效type错误")
                    all_passed.append("multipath_invalid_type_handling")
                
            except Exception as e:
                print(f"❌ MultiPath错误处理测试失败: {e}")
                all_failed.append("multipath_error_handling_overall")
            
            # === 测试 4: MultiPath检索器动态管理 ===
            print("\n=== 测试 4: MultiPath检索器动态管理 ===")
            try:
                config_data = {
                    "documents": [
                        {
                            "id": doc.id,
                            "content": doc.content,
                            "metadata": doc.metadata
                        }
                        for doc in documents
                    ],
                    "query": "machine learning",
                    "multipath_config": {
                        "type": "multipath",
                        "k": 3,
                        "with_score": True,
                        "top_k_per_retriever": 10,
                        "search_kwargs": {},
                        "indexers": [
                            {
                                "type": "bm25_indexer",
                                "index_path": os.path.join(temp_dir, "multipath_dynamic_bm25"),
                                "bm25_k1": 1.2,
                                "bm25_b": 0.75
                            }
                        ]
                    }
                }
                
                # 使用model_validate直接从JSON数据创建MultiPathRetrieverConfig对象
                multipath_config = MultiPathRetrieverConfig.model_validate(config_data.get("multipath_config", {}))
                
                # 从multipath_config中获取bm25_config
                bm25_config = None
                for indexer in multipath_config.indexers:
                    if indexer.type == "bm25_indexer":
                        bm25_config = indexer
                
                # 确保索引目录存在
                bm25_dir = bm25_config.index_path
                os.makedirs(bm25_dir, exist_ok=True)
                
                # 构建并保存 BM25 索引
                bm25_builder = bm25_config.build()
                bm25_builder.from_documents(documents)
                
                # 初始化多路检索器
                mp = multipath_config.build()
                
                # 执行检索（只有BM25）
                query = config_data["query"]
                results = mp.invoke(query)
                
                if len(results) > 0:
                    print(f"✅ 单索引器MultiPath返回 {len(results)} 个结果")
                    all_passed.append("multipath_single_indexer_works")
                else:
                    print("❌ 单索引器MultiPath未返回结果")
                    all_failed.append("multipath_single_indexer_works")
                
                # 清理资源
                bm25_builder.close()
                
            except Exception as e:
                print(f"❌ MultiPath动态管理测试失败: {e}")
                all_failed.append("multipath_dynamic_management_overall")
    
    except Exception as e:
        print(f"❌ MultiPath综合测试失败: {e}")
        all_failed.append("multipath_comprehensive_test_overall")
    
    # 测试总结
    print("\n" + "=" * 60)
    print("📊 MultiPath检索测试总结")
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


def run_all_retrieval_tests():
    """运行所有检索器测试"""
    print("🚀 开始全面检索测试套件")
    print("=" * 80)
    
    all_tests = [
        ("BM25检索器", test_comprehensive_retrieval),
        ("Dense检索器", test_dense_retrieval), 
        ("MultiPath检索器", test_multipath_retrieval)
    ]
    
    results = {}
    overall_success = True
    
    for test_name, test_func in all_tests:
        print(f"\n{'=' * 20} {test_name}测试 {'=' * 20}")
        try:
            success = test_func()
            results[test_name] = success
            if not success:
                overall_success = False
        except Exception as e:
            print(f"❌ {test_name}测试出现意外错误: {e}")
            results[test_name] = False
            overall_success = False
    
    # 总体测试结果
    print("\n" + "=" * 80)
    print("📊 全面检索测试总结")
    print("=" * 80)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"\n总体成功率: {success_rate:.1f}% ({success_count}/{total_count})")
    
    if overall_success:
        print("\n🎉 所有检索器测试都通过了！")
    else:
        print("\n❌ 部分检索器测试失败。请检查上面的详细输出。")
    
    print("=" * 80)
    
    return overall_success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='运行检索器测试')
    parser.add_argument('--test', choices=['all', 'bm25', 'dense', 'multipath'], 
                       default='all', help='选择要运行的测试类型')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_retrieval_tests()
    elif args.test == 'bm25':
        print('开始BM25检索测试')
        success = test_comprehensive_retrieval()
    elif args.test == 'dense':
        print('开始Dense检索测试')
        success = test_dense_retrieval()
    elif args.test == 'multipath':
        print('开始MultiPath检索测试')
        success = test_multipath_retrieval()
    
    if success:
        print("\n🎉 选定的测试都通过了！")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败。请检查上面的输出。")
        sys.exit(1)
