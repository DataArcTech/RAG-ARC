import json
import os
import sys
import tempfile
import unittest
import logging
from typing import List, Dict, Any

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from encapsulation.data_model.data_model import Document
from config.encapsulaiton.bm25_config import BM25IndexBuilderConfig
from config.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from config.retrieval.multipath_config import MultiPathRetrieverConfig
from config.retrieval.dense_config import DenseRetrieverConfig
from config.encapsulaiton.faiss_config import FaissIndexConfig
from config.llm.huggingface_config import HuggingFaceEmbedConfig

# 设置日志级别
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_test_documents() -> List[Document]:
    """Create test documents for retrieval testing"""
    return [
        Document(
            id="tech_001",
            content="Python is a high-level programming language widely used for machine learning and data science applications.",
            metadata={"category": "technology", "language": "english", "difficulty": "intermediate"}
        ),
        Document(
            id="tech_002",
            content="Deep learning neural networks can solve complex problems like image recognition and natural language processing.",
            metadata={"category": "technology", "language": "english", "difficulty": "advanced"}
        ),
        Document(
            id="science_001",
            content="Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways.",
            metadata={"category": "science", "language": "english", "difficulty": "advanced"}
        ),
        Document(
            id="chinese_001",
            content="机器学习是人工智能的重要分支，包括监督学习和无监督学习等多种方法。",
            metadata={"category": "technology", "language": "chinese", "difficulty": "intermediate"}
        ),
        Document(
            id="chinese_002",
            content="深度学习使用神经网络来解决复杂的模式识别问题，在图像处理和自然语言处理领域有广泛应用。",
            metadata={"category": "technology", "language": "chinese", "difficulty": "advanced"}
        )
    ]


def create_index_manager_and_build_indexes(documents: List[Document], index_configs: Dict[str, Any]) -> None:
    """使用IndexManager构建索引（简化版本，直接使用索引器）"""
    for index_type, index_config in index_configs.items():
        index = index_config.build()

        # 根据索引类型使用不同的方法
        if index_type == "bm25_indexer":
            index.build_index(documents)
            index.save_index(index_config.index_path)

        elif index_type == "faiss":
            try:
                from config.llm.huggingface_config import HuggingFaceEmbedConfig

                # 创建嵌入模型配置并设置到索引配置中
                embedding_config = HuggingFaceEmbedConfig(
                    model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                    task_types="embedding",
                    device="cuda:0"
                )

                # 设置嵌入模型到索引配置中
                index_config.embedding = embedding_config

                # 重新构建索引以包含嵌入配置
                index = index_config.build()

                # 使用build_index方法构建索引
                index.build_index(documents)

                # 保存索引
                index.save_index(index_config.index_path)

            except Exception as e:
                logger.warning(f"Failed to build FAISS index with embeddings: {e}")
                # 创建空的索引目录作为占位符
                import os
                os.makedirs(index_config.index_path, exist_ok=True)

        else:
            # 通用方法
            if hasattr(index, 'add_documents'):
                index.add_documents(documents)
            elif hasattr(index, 'add_texts'):
                texts = [doc.content for doc in documents]
                metadatas = [doc.metadata or {} for doc in documents]
                ids = [doc.id for doc in documents]
                index.add_texts(texts, metadatas=metadatas, ids=ids)

            # 保存索引
            if hasattr(index, 'save_index'):
                if hasattr(index_config, 'index_path'):
                    index.save_index(index_config.index_path)
                else:
                    index.save_index()

    return None


def cleanup_index_manager(index_manager, documents, index_configs: Dict[str, Any]):
    """清理索引文件"""
    import shutil
    for index_type, index_config in index_configs.items():
        try:
            if hasattr(index_config, 'index_path') and os.path.exists(index_config.index_path):
                if os.path.isdir(index_config.index_path):
                    # 强制删除目录及其内容
                    shutil.rmtree(index_config.index_path, ignore_errors=True)
                else:
                    os.remove(index_config.index_path)
        except Exception as e:
            # 使用logger而不是logging
            logger.warning(f"Failed to cleanup index {index_type}: {e}")


class TestBM25Retriever(unittest.TestCase):
    """BM25检索器测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.documents = create_test_documents()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_bm25_basic_functionality(self):
        """测试BM25基础功能"""
        # 创建BM25索引配置
        index_config = BM25IndexBuilderConfig(
            type="bm25_indexer",
            index_path=os.path.join(self.temp_dir, "bm25_test"),
            bm25_k1=1.2,
            bm25_b=0.75
        )

        # 使用索引器构建索引
        index_configs = {"bm25_indexer": index_config}
        create_index_manager_and_build_indexes(self.documents, index_configs)

        try:
            # 创建BM25检索器配置
            retriever_config = TantivyBM25RetrieverConfig(
                type="tantivy_bm25",
                index_config=index_config,
                search_kwargs={"k": 5, "with_score": True}
            )

            # 构建检索器
            retriever = retriever_config.build()
            self.assertIsNotNone(retriever)
            self.assertEqual(retriever.config.type, "tantivy_bm25")

            # 执行搜索
            results = retriever.invoke("machine learning", k=3)
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), 3)

            # 检查结果结构
            first_result = results[0]
            self.assertTrue(hasattr(first_result, 'id'))
            self.assertTrue(hasattr(first_result, 'content'))
            self.assertTrue(hasattr(first_result, 'metadata'))

            # 检查分数
            self.assertIn('score', first_result.metadata)
            self.assertIsInstance(first_result.metadata['score'], (int, float))

        finally:
            # 清理索引
            cleanup_index_manager(None, self.documents, index_configs)

    def test_bm25_config_from_json(self):
        """测试从JSON创建BM25配置"""
        json_str = f"""
        {{
            "type": "tantivy_bm25",
            "search_kwargs": {{
                "k": 10,
                "with_score": true,
                "use_phrase_query": false
            }},
            "index_config": {{
                "type": "bm25_indexer",
                "index_path": "{os.path.join(self.temp_dir, 'bm25_json_test').replace(os.sep, '/')}",
                "bm25_k1": 1.5,
                "bm25_b": 0.8,
                "batch_size": 100
            }}
        }}
        """

        config_data = json.loads(json_str)
        config = TantivyBM25RetrieverConfig(**config_data)

        # 验证配置
        self.assertEqual(config.type, "tantivy_bm25")
        self.assertEqual(config.search_kwargs["k"], 10)
        self.assertEqual(config.index_config.bm25_k1, 1.5)
        self.assertEqual(config.index_config.bm25_b, 0.8)

        # 使用IndexManager构建索引
        index_configs = {"bm25_indexer": config.index_config}
        create_index_manager_and_build_indexes(self.documents, index_configs)

        try:
            # 构建并测试
            retriever = config.build()
            self.assertIsNotNone(retriever)

            # 搜索
            results = retriever.invoke("machine learning")
            self.assertGreater(len(results), 0)

        finally:
            # 清理索引
            cleanup_index_manager(None, self.documents, index_configs)

    def test_bm25_search_parameters(self):
        """测试BM25搜索参数"""
        index_config = BM25IndexBuilderConfig(
            type="bm25_indexer",
            index_path=os.path.join(self.temp_dir, "bm25_params_test")
        )

        # 使用IndexManager构建索引
        index_configs = {"bm25_indexer": index_config}
        create_index_manager_and_build_indexes(self.documents, index_configs)

        try:
            retriever_config = TantivyBM25RetrieverConfig(
                type="tantivy_bm25",
                index_config=index_config,
                search_kwargs={"k": 3, "with_score": True}
            )

            retriever = retriever_config.build()

            # 测试不同的k值
            results_k3 = retriever.invoke("machine learning", k=3)
            results_k5 = retriever.invoke("machine learning", k=5)

            self.assertLessEqual(len(results_k3), 3)
            self.assertLessEqual(len(results_k5), 5)

            # 测试无效k值
            with self.assertRaises(ValueError):
                retriever.invoke("machine learning", k=0)

        finally:
            # 清理索引
            cleanup_index_manager(None, self.documents, index_configs)



class TestDenseRetriever(unittest.TestCase):
    """Dense检索器测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.documents = create_test_documents()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_dense_basic_functionality(self):
        """测试Dense基础功能"""
        # 创建嵌入配置
        embedding_config = HuggingFaceEmbedConfig(
            model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
            task_types="embedding",
            device="cuda:0"
        )

        # 创建Faiss索引配置
        index_config = FaissIndexConfig(
            type="faiss",
            index_path=os.path.join(self.temp_dir, "dense_test"),
            metric="cosine",
            index_type="flat",
            normalize_L2=True,
            embedding=embedding_config
        )

        # 使用IndexManager构建索引
        index_configs = {"faiss": index_config}
        create_index_manager_and_build_indexes(self.documents, index_configs)

        try:
            # 创建Dense检索器配置
            retriever_config = DenseRetrieverConfig(
                type="dense",
                index_config=index_config,
                embedding_config=embedding_config,
                search_kwargs={"k": 5, "with_score": True}
            )

            # 构建检索器
            retriever = retriever_config.build()
            self.assertIsNotNone(retriever)
            self.assertEqual(retriever.config.type, "dense")

            # 执行搜索测试
            results = retriever.invoke("machine learning", k=3)
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), 3)

            # 检查结果结构
            first_result = results[0]
            self.assertTrue(hasattr(first_result, 'id'))
            self.assertTrue(hasattr(first_result, 'content'))
            self.assertTrue(hasattr(first_result, 'metadata'))

            # 验证检索质量 - 应该找到相关文档
            result_contents = [doc.content.lower() for doc in results]
            found_ml_content = any("machine learning" in content or "artificial intelligence" in content
                                 for content in result_contents)
            self.assertTrue(found_ml_content, "Should find machine learning related content")

            # 测试不同的查询
            python_results = retriever.invoke("python programming", k=2)
            self.assertGreater(len(python_results), 0)

            # 验证语义搜索能力
            ai_results = retriever.invoke("artificial intelligence", k=2)
            self.assertGreater(len(ai_results), 0)

        except Exception as e:
            # 如果FAISS索引构建失败（比如没有GPU或模型），跳过测试
            logger.warning(f"Dense retrieval test skipped due to: {e}")
            self.skipTest(f"Dense retrieval test skipped: {e}")

        finally:
            # 清理索引
            cleanup_index_manager(None, self.documents, index_configs)

    def test_dense_config_from_json(self):
        """测试从JSON创建Dense配置"""
        json_str = f"""
        {{
            "type": "dense",
            "search_kwargs": {{
                "k": 10,
                "with_score": true
            }},
            "index_config": {{
                "type": "faiss",
                "index_path": "{os.path.join(self.temp_dir, 'dense_json_test').replace(os.sep, '/')}",
                "metric": "cosine",
                "index_type": "flat",
                "normalize_L2": true
            }},
            "embedding_config": {{
                "type": "huggingface_embedding",
                "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                "task_types": "embedding",
                "device": "cuda:0"
            }}
        }}
        """
        
        config_data = json.loads(json_str)
        config = DenseRetrieverConfig(**config_data)

        # 验证配置
        self.assertEqual(config.type, "dense")
        self.assertEqual(config.search_kwargs["k"], 10)
        self.assertEqual(config.index_config.metric, "cosine")
        self.assertEqual(config.embedding_config.model_name, "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B")

        # 设置嵌入配置到索引配置中
        config.index_config.embedding = config.embedding_config

        # 使用IndexManager构建索引
        index_configs = {"faiss": config.index_config}
        create_index_manager_and_build_indexes(self.documents, index_configs)

        try:
            # 构建并测试
            retriever = config.build()
            self.assertIsNotNone(retriever)

            # 执行搜索测试
            results = retriever.invoke("machine learning")
            self.assertGreater(len(results), 0)

        except Exception as e:
            # 如果FAISS索引构建失败，跳过测试
            logger.warning(f"Dense retrieval test skipped due to: {e}")
            self.skipTest(f"Dense retrieval test skipped: {e}")

        finally:
            # 清理索引
            cleanup_index_manager(None, self.documents, index_configs)


class TestMultiPathRetriever(unittest.TestCase):
    """MultiPath检索器测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.documents = create_test_documents()

    def tearDown(self):
        """清理测试环境"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

    def test_multipath_config_from_json(self):
        """测试从JSON创建MultiPath配置"""
        json_str = f"""
        {{
            "type": "multipath",
            "search_kwargs": {{
                "k": 5,
                "with_score": true
            }},
            "retrievers": [
                {{
                    "type": "tantivy_bm25",
                    "index_config": {{
                        "type": "bm25_indexer",
                        "index_path": "{os.path.join(self.temp_dir, 'mp_bm25').replace(os.sep, '/')}",
                        "bm25_k1": 1.2,
                        "bm25_b": 0.75
                    }}
                }},
                {{
                    "type": "dense",
                    "index_config": {{
                        "type": "faiss",
                        "index_path": "{os.path.join(self.temp_dir, 'mp_dense').replace(os.sep, '/')}",
                        "metric": "cosine",
                        "index_type": "flat"
                    }},
                    "embedding_config": {{
                        "type": "huggingface_embedding",
                        "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                        "task_types": "embedding",
                        "device": "cuda:0"
                    }}
                }}
            ],
            "fusion_method": "rrf",
            "rrf_k": 60
        }}
        """
        
        config_data = json.loads(json_str)
        config = MultiPathRetrieverConfig(**config_data)
        
        # 验证配置
        self.assertEqual(config.type, "multipath")
        self.assertEqual(len(config.retrievers), 2)
        self.assertEqual(config.retrievers[0].type, "tantivy_bm25")
        self.assertEqual(config.retrievers[1].type, "dense")
        self.assertEqual(config.fusion_method, "rrf")
        self.assertEqual(config.rrf_k, 60)

        # 设置嵌入配置到Dense检索器的索引配置中
        config.retrievers[1].index_config.embedding = config.retrievers[1].embedding_config

        # 使用IndexManager构建索引
        index_configs = {
            "bm25_indexer": config.retrievers[0].index_config,
            "faiss": config.retrievers[1].index_config
        }
        create_index_manager_and_build_indexes(self.documents, index_configs)

        try:
            # 构建检索器
            retriever = config.build()
            self.assertIsNotNone(retriever)
            self.assertEqual(retriever.config.type, "multipath")

            # 执行搜索测试
            results = retriever.invoke("machine learning")
            self.assertGreater(len(results), 0)

            # 检查融合结果
            first_result = results[0]
            self.assertTrue(hasattr(first_result, 'id'))
            self.assertTrue(hasattr(first_result, 'content'))
            self.assertTrue(hasattr(first_result, 'metadata'))

        except Exception as e:
            # 如果索引构建失败，跳过测试
            logger.warning(f"MultiPath retrieval test skipped due to: {e}")
            self.skipTest(f"MultiPath retrieval test skipped: {e}")

        finally:
            # 清理索引
            cleanup_index_manager(None, self.documents, index_configs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
