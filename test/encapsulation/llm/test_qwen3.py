import unittest
import json
import os
import sys
from typing import Dict, Any, List
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from framework.register import Register
from encapsulation.llm.qwen3 import QwenConfig, QwenLLM
from encapsulation.data_model.data_model import Document


class TestQwenRerankerWithRegister(unittest.TestCase):
    """测试使用注册器的Qwen重排序模型功能"""
    
    def setUp(self):
        """测试前的设置"""
        self.register = Register()
        self.config_path = os.path.join(os.path.dirname(__file__), 'qwen_reranker_test_config.json')
        self.app_name = "test_qwen_reranker"
        
        # 创建临时缓存目录
        self.temp_cache_dir = tempfile.mkdtemp()
        
        # 确保配置文件存在
        self._create_test_config()
        
        # 创建测试文档
        self.test_documents = [
            Document(content="苹果是一种水果，富含维生素和纤维，味道甜美。"),
            Document(content="猫咪是可爱的宠物，需要主人的关爱和照料。"),
            Document(content="Python是一种编程语言，广泛用于数据科学和机器学习。"),
            Document(content="橙子含有丰富的维生素C，对免疫系统有益。"),
            Document(content="机器学习是人工智能的一个重要分支。")
        ]
    
    def _create_test_config(self):
        """创建测试配置文件"""
        config_data = {
            "type": "qwen_reranker",
            "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B",  # 使用轻量级模型进行测试
            "task_types": "rerank",
            "device": "cuda:0",
            "cache_folder": self.temp_cache_dir,
            "model_kwargs": {
                "torch_dtype": "float16"
            },
            "instruction": "Given the user query, retrieve the relevant passages"
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def tearDown(self):
        """测试后的清理"""
        # 清理注册器
        if hasattr(self.register, 'registrations'):
            self.register.registrations.clear()
        
        # 删除测试配置文件
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        
        # 清理临时缓存目录
        if os.path.exists(self.temp_cache_dir):
            shutil.rmtree(self.temp_cache_dir)
    
    @classmethod
    def setUpClass(cls):
        """类级别设置 - 检查依赖"""
        try:
            import transformers
            import torch
            cls.has_transformers = True
            cls.has_cuda = torch.cuda.is_available()
            if not cls.has_cuda:
                print("\n警告: CUDA 不可用，将使用CPU进行测试")
        except ImportError:
            cls.has_transformers = False
            print("\n警告: transformers 未安装，将跳过相关测试")
    
    def test_register_qwen_reranker(self):
        """测试注册Qwen重排序模型"""
        print("\n=== 测试注册Qwen重排序模型 ===")
        
        if not self.has_transformers:
            self.skipTest("transformers not installed")
        
        # 如果没有CUDA，修改配置使用CPU
        if not self.has_cuda:
            config_data = json.load(open(self.config_path))
            config_data["device"] = "cpu"
            config_data["model_kwargs"]["torch_dtype"] = "float32"
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f)
        
        # 注册Qwen重排序模型
        self.register.register(self.config_path, self.app_name, QwenConfig)
        
        # 验证注册成功
        self.assertIn(self.app_name, self.register.registrations)
        
        # 获取注册的对象
        llm_instance = self.register.get_object(self.app_name)
        
        # 验证对象类型
        self.assertIsInstance(llm_instance, QwenLLM)
        
        # 验证配置
        self.assertEqual(llm_instance.config.model_name, "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B")
        self.assertEqual(llm_instance.config.task_types, "rerank")
        
        print(f"✓ 成功注册Qwen重排序模型: {llm_instance}")
        print(f"✓ 模型名称: {llm_instance.config.model_name}")
        print(f"✓ 支持的任务类型: {llm_instance.config.task_types}")
        print(f"✓ 使用设备: {llm_instance.config.device}")
    
    def test_reranking_functionality(self):
        """测试重排序功能"""
        print("\n=== 测试重排序功能 ===")
        
        if not self.has_transformers:
            self.skipTest("transformers not installed")
        
        # 如果没有CUDA，修改配置使用CPU
        if not self.has_cuda:
            config_data = json.load(open(self.config_path))
            config_data["device"] = "cpu"
            config_data["model_kwargs"]["torch_dtype"] = "float32"
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f)
        
        # 注册重排序模型
        self.register.register(self.config_path, self.app_name, QwenConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试基本重排序功能
        query = "水果的营养价值"
        ranked_docs = llm_instance._rerank(query, self.test_documents)
        
        # 验证返回结果格式
        self.assertIsInstance(ranked_docs, list)
        self.assertGreater(len(ranked_docs), 0)
        
        for doc_index, score in ranked_docs:
            self.assertIsInstance(doc_index, int)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(doc_index, 0)
            self.assertLess(doc_index, len(self.test_documents))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # 验证结果按分数降序排列
        scores = [score for _, score in ranked_docs]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        print("✓ 基本重排序功能测试通过")
        print(f"✓ 返回文档数量: {len(ranked_docs)}")
        print(f"✓ 最高分数: {scores[0]:.4f}")
        print(f"✓ 最低分数: {scores[-1]:.4f}")
    
    def test_reranking_with_top_k(self):
        """测试带top_k参数的重排序功能"""
        print("\n=== 测试top_k重排序功能 ===")
        
        if not self.has_transformers:
            self.skipTest("transformers not installed")
        
        # 如果没有CUDA，修改配置使用CPU
        if not self.has_cuda:
            config_data = json.load(open(self.config_path))
            config_data["device"] = "cpu"
            config_data["model_kwargs"]["torch_dtype"] = "float32"
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f)
        
        # 注册重排序模型
        self.register.register(self.config_path, self.app_name, QwenConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试top_k=2的重排序
        query = "编程语言"
        top_k = 2
        ranked_docs = llm_instance._rerank(query, self.test_documents, top_k=top_k)
        
        # 验证返回的文档数量
        self.assertEqual(len(ranked_docs), top_k)
        
        # 验证结果仍然按分数降序排列
        scores = [score for _, score in ranked_docs]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        print(f"✓ top_k={top_k}重排序功能测试通过")
        print(f"✓ 返回文档数量: {len(ranked_docs)}")
        
        # 测试top_k大于文档数量的情况
        large_k = len(self.test_documents) + 5
        ranked_docs_large = llm_instance._rerank(query, self.test_documents, top_k=large_k)
        self.assertEqual(len(ranked_docs_large), len(self.test_documents))
        
        print(f"✓ top_k超出文档数量时的处理正常")
    
    def test_compute_scores_method(self):
        """测试compute_scores方法"""
        print("\n=== 测试compute_scores方法 ===")
        
        if not self.has_transformers:
            self.skipTest("transformers not installed")
        
        # 如果没有CUDA，修改配置使用CPU
        if not self.has_cuda:
            config_data = json.load(open(self.config_path))
            config_data["device"] = "cpu"
            config_data["model_kwargs"]["torch_dtype"] = "float32"
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f)
        
        # 注册重排序模型
        self.register.register(self.config_path, self.app_name, QwenConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 创建query-document对
        query = "水果营养"
        pairs = [(query, doc.content) for doc in self.test_documents[:3]]
        
        # 计算分数
        scores = llm_instance.compute_scores(pairs)
        
        # 验证分数格式
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 3)
        
        for score in scores:
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        print("✓ compute_scores方法测试通过")
        print(f"✓ 计算的分数: {[f'{score:.4f}' for score in scores]}")
    
    def test_model_info(self):
        """测试模型信息获取"""
        print("\n=== 测试模型信息获取 ===")
        
        if not self.has_transformers:
            self.skipTest("transformers not installed")
        
        # 如果没有CUDA，修改配置使用CPU
        if not self.has_cuda:
            config_data = json.load(open(self.config_path))
            config_data["device"] = "cpu"
            config_data["model_kwargs"]["torch_dtype"] = "float32"
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f)
        
        # 注册重排序模型
        self.register.register(self.config_path, self.app_name, QwenConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 获取模型信息
        model_info = llm_instance.get_model_info()
        
        # 验证信息完整性
        required_fields = ["model_name", "task_types", "device", "model_type"]
        for field in required_fields:
            self.assertIn(field, model_info)
        
        self.assertEqual(model_info["model_name"], "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B")
        self.assertEqual(model_info["task_types"], "rerank")
        
        print("✓ 模型信息获取测试通过")
        print(f"✓ 模型信息: {model_info}")
    
    def test_unsupported_methods(self):
        """测试不支持的方法"""
        print("\n=== 测试不支持的方法 ===")
        
        if not self.has_transformers:
            self.skipTest("transformers not installed")
        
        # 如果没有CUDA，修改配置使用CPU
        if not self.has_cuda:
            config_data = json.load(open(self.config_path))
            config_data["device"] = "cpu"
            config_data["model_kwargs"]["torch_dtype"] = "float32"
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f)
        
        # 注册重排序模型
        self.register.register(self.config_path, self.app_name, QwenConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试不支持的聊天功能
        messages = [{"role": "user", "content": "测试消息"}]
        with self.assertRaises(NotImplementedError) as cm:
            llm_instance._chat(messages)
        self.assertIn("do not support chat", str(cm.exception))
        
        # 测试不支持的流式聊天功能
        with self.assertRaises(NotImplementedError) as cm:
            llm_instance._stream_chat(messages)
        self.assertIn("do not support streaming chat", str(cm.exception))
        
        # 测试不支持的嵌入功能
        with self.assertRaises(NotImplementedError) as cm:
            llm_instance._embed(["测试文本"])
        self.assertIn("do not support embedding", str(cm.exception))
        
        print("✓ 不支持的聊天功能测试通过")
        print("✓ 不支持的流式聊天功能测试通过")
        print("✓ 不支持的嵌入功能测试通过")
    
    def test_relevance_scoring(self):
        """测试相关性评分的合理性"""
        print("\n=== 测试相关性评分合理性 ===")
        
        if not self.has_transformers:
            self.skipTest("transformers not installed")
        
        # 如果没有CUDA，修改配置使用CPU
        if not self.has_cuda:
            config_data = json.load(open(self.config_path))
            config_data["device"] = "cpu"
            config_data["model_kwargs"]["torch_dtype"] = "float32"
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f)
        
        # 注册重排序模型
        self.register.register(self.config_path, self.app_name, QwenConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试明显相关和不相关的查询
        relevant_query = "水果维生素"
        ranked_docs = llm_instance._rerank(relevant_query, self.test_documents)
        
        # 找到包含水果内容的文档索引
        fruit_doc_indices = set()
        for i, doc in enumerate(self.test_documents):
            if "水果" in doc.content or "苹果" in doc.content or "橙子" in doc.content:
                fruit_doc_indices.add(i)
        
        # 验证相关文档的排名
        if fruit_doc_indices:
            top_ranked_indices = {idx for idx, _ in ranked_docs[:2]}  # 前2个结果
            # 至少有一个水果相关的文档应该排在前面
            self.assertTrue(len(fruit_doc_indices.intersection(top_ranked_indices)) > 0,
                          "相关文档应该排在前面")
        
        print("✓ 相关性评分合理性测试通过")
        
        # 显示排序结果以便验证
        print("查询: '水果维生素'")
        for i, (doc_idx, score) in enumerate(ranked_docs[:3]):
            content_preview = self.test_documents[doc_idx].content[:20] + "..."
            print(f"  排名 {i+1}: 分数 {score:.4f} - {content_preview}")


def run_tests():
    """运行所有测试"""
    print("开始运行Qwen重排序模型注册器测试...")
    print("=" * 60)
    
    # 检查依赖
    try:
        import transformers
        import torch
        print("✓ transformers 已安装")
        if torch.cuda.is_available():
            print("✓ CUDA 可用")
        else:
            print("⚠ CUDA 不可用，将使用CPU进行测试")
    except ImportError:
        print("⚠ transformers 未安装，部分测试将被跳过")
        print("安装命令: pip install transformers torch")
    
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestQwenRerankerWithRegister)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"运行测试: {result.testsRun}")
    print(f"跳过测试: {len([test for test in result.skipped]) if hasattr(result, 'skipped') else 0}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if hasattr(result, 'skipped') and result.skipped:
        print(f"\n跳过的测试:")
        for test, reason in result.skipped:
            print(f"- {test}: {reason}")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)