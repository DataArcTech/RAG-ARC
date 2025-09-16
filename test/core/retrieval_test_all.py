import json
import os
import sys
import tempfile
import unittest
import logging
from typing import List

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.utils.data_model import Document
from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig
from core.retrieval.tantivy_bm25 import TantivyBM25RetrieverConfig
from core.retrieval.multipath import MultiPathRetrieverConfig
from core.retrieval.dense import DenseRetrieverConfig
from encapsulation.database.vector_db.faiss import FaissIndexConfig
from encapsulation.llm.huggingface import HuggingFaceEmbedConfig

# 设置日志级别
logging.basicConfig(level=logging.WARNING)


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
        
        # 通过检索器添加文档（这样会正确处理索引）
        doc_ids = retriever.add_documents(self.documents)
        self.assertEqual(len(doc_ids), len(self.documents))
        
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
        
        # 构建并测试
        retriever = config.build()
        self.assertIsNotNone(retriever)
        
        # 添加文档并搜索
        retriever.add_documents(self.documents)
        results = retriever.invoke("machine learning")
        self.assertGreater(len(results), 0)

    def test_bm25_search_parameters(self):
        """测试BM25搜索参数"""
        index_config = BM25IndexBuilderConfig(
            type="bm25_indexer",
            index_path=os.path.join(self.temp_dir, "bm25_params_test")
        )
        
        retriever_config = TantivyBM25RetrieverConfig(
            type="tantivy_bm25",
            index_config=index_config,
            search_kwargs={"k": 3, "with_score": True}
        )
        
        retriever = retriever_config.build()
        retriever.add_documents(self.documents)
        
        # 测试不同的k值
        results_k3 = retriever.invoke("machine learning", k=3)
        results_k5 = retriever.invoke("machine learning", k=5)
        
        self.assertLessEqual(len(results_k3), 3)
        self.assertLessEqual(len(results_k5), 5)
        
        # 测试无效k值
        with self.assertRaises(ValueError):
            retriever.invoke("machine learning", k=0)

    def test_bm25_multilingual(self):
        """测试BM25多语言支持"""
        index_config = BM25IndexBuilderConfig(
            type="bm25_indexer",
            index_path=os.path.join(self.temp_dir, "bm25_multilingual_test")
        )
        
        retriever_config = TantivyBM25RetrieverConfig(
            type="tantivy_bm25",
            index_config=index_config
        )
        
        retriever = retriever_config.build()
        retriever.add_documents(self.documents)
        
        # 英文搜索
        english_results = retriever.invoke("machine learning")
        self.assertGreater(len(english_results), 0)
        
        # 中文搜索
        chinese_results = retriever.invoke("机器学习")
        self.assertGreater(len(chinese_results), 0)


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
            normalize_L2=True
        )
        
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
        
        # 通过检索器添加文档
        doc_ids = retriever.add_documents(self.documents)
        self.assertEqual(len(doc_ids), len(self.documents))
        
        # 执行搜索
        results = retriever.invoke("machine learning", k=3)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        
        # 检查结果结构
        first_result = results[0]
        self.assertTrue(hasattr(first_result, 'id'))
        self.assertTrue(hasattr(first_result, 'content'))
        self.assertTrue(hasattr(first_result, 'metadata'))

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
        
        # 构建并测试
        retriever = config.build()
        self.assertIsNotNone(retriever)


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
        
        # 构建检索器
        retriever = config.build()
        self.assertIsNotNone(retriever)
        self.assertEqual(retriever.config.type, "multipath")


if __name__ == "__main__":
    unittest.main(verbosity=2)
