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
from encapsulation.llm.huggingface import HuggingFaceEmbedConfig, HuggingFaceEmbed


class TestHuggingFaceWithRegister(unittest.TestCase):
    """测试使用注册器的HuggingFace嵌入模型功能"""
    
    def setUp(self):
        """测试前的设置"""
        self.register = Register()
        self.config_path = os.path.join(os.path.dirname(__file__), 'huggingface_test_config.json')
        self.app_name = "test_huggingface_embed"
        
        # 创建临时缓存目录
        self.temp_cache_dir = tempfile.mkdtemp()
        
        # 确保配置文件存在
        self._create_test_config()
    
    def _create_test_config(self):
        """创建测试配置文件"""
        config_data = {
            "type": "huggingface_embedding",
            "model_name": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",  # 轻量级模型，适合测试
            "task_types": "embedding",
            "device": "cuda:0",
            "cache_folder": self.temp_cache_dir,
            "model_kwargs": {
                "trust_remote_code": False
            },
            "encode_kwargs": {
                "normalize_embeddings": True,
                "batch_size": 32,
                "show_progress_bar": False
            }
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
            import sentence_transformers
            cls.has_sentence_transformers = True
        except ImportError:
            cls.has_sentence_transformers = False
            print("\n警告: sentence-transformers 未安装，将跳过相关测试")
    
    def test_register_huggingface_embed(self):
        """测试注册HuggingFace嵌入模型"""
        print("\n=== 测试注册HuggingFace嵌入模型 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册HuggingFace嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        
        # 验证注册成功
        self.assertIn(self.app_name, self.register.registrations)
        
        # 获取注册的对象
        llm_instance = self.register.get_object(self.app_name)
        
        # 验证对象类型
        self.assertIsInstance(llm_instance, HuggingFaceEmbed)
        
        # 验证配置
        self.assertEqual(llm_instance.config.model_name, "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B")
        self.assertEqual(llm_instance.config.task_types, "embedding")
        self.assertEqual(llm_instance.config.device, "cuda:0")
        
        print(f"✓ 成功注册HuggingFace嵌入模型: {llm_instance}")
        print(f"✓ 模型名称: {llm_instance.config.model_name}")
        print(f"✓ 支持的任务类型: {llm_instance.config.task_types}")
        print(f"✓ 使用设备: {llm_instance.config.device}")
    
    def test_embedding_functionality(self):
        """测试嵌入功能"""
        print("\n=== 测试嵌入功能 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试单个文本嵌入
        text = "这是一个测试文本"
        embedding = llm_instance.embed_query(text)
        
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)
        
        expected_dim = 1024
        self.assertEqual(len(embedding), expected_dim)
        
        # 测试多个文本嵌入
        texts = ["文本1", "文本2", "文本3"]
        embeddings = llm_instance.embed_documents(texts)
        
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), 3)
        for emb in embeddings:
            self.assertIsInstance(emb, list)
            self.assertEqual(len(emb), expected_dim)
            self.assertIsInstance(emb[0], float)
        
        # 测试嵌入向量的数值范围（归一化后应该在合理范围内）
        import math
        embedding_norm = math.sqrt(sum(x*x for x in embedding))
        self.assertAlmostEqual(embedding_norm, 1.0, places=5)  # 归一化后的向量模长应该接近1
        
        print("✓ 单个文本嵌入功能测试通过")
        print(f"✓ 嵌入维度: {len(embedding)}")
        print(f"✓ 嵌入向量模长: {embedding_norm:.6f}")
        print("✓ 多个文本嵌入功能测试通过")
    
    def test_private_embed_method(self):
        """测试私有_embed方法"""
        print("\n=== 测试私有_embed方法 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试单个文本
        single_result = llm_instance._embed("单个文本")
        self.assertIsInstance(single_result, list)
        self.assertEqual(len(single_result), 1024)
        self.assertIsInstance(single_result[0], float)
        
        # 测试文本列表
        list_result = llm_instance._embed(["文本1", "文本2"])
        self.assertIsInstance(list_result, list)
        self.assertEqual(len(list_result), 2)
        for emb in list_result:
            self.assertIsInstance(emb, list)
            self.assertEqual(len(emb), 1024)
        
        print("✓ 私有_embed方法单个文本测试通过")
        print("✓ 私有_embed方法文本列表测试通过")
    
    def test_model_info(self):
        """测试模型信息获取"""
        print("\n=== 测试模型信息获取 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 获取模型信息
        model_info = llm_instance.get_model_info()
        
        # 验证信息完整性
        self.assertIn("model_name", model_info)
        self.assertIn("task_types", model_info)
        self.assertIn("device", model_info)
        self.assertIn("cache_folder", model_info)
        self.assertIn("provider", model_info)
        self.assertIn("model_type", model_info)
        
        self.assertEqual(model_info["model_name"], "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B")
        self.assertEqual(model_info["task_types"], "embedding")
        self.assertEqual(model_info["device"], "cuda:0")
        self.assertEqual(model_info["provider"], "huggingface")
        self.assertEqual(model_info["model_type"], "sentence_transformer")
        self.assertEqual(model_info["cache_folder"], self.temp_cache_dir)
        
        print("✓ 模型信息获取测试通过")
        print(f"✓ 模型信息: {model_info}")
    
    def test_text_preprocessing(self):
        """测试文本预处理功能"""
        print("\n=== 测试文本预处理功能 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试包含换行符的文本
        text_with_newlines = "这是第一行\n这是第二行\n这是第三行"
        embedding = llm_instance.embed_documents([text_with_newlines])
        
        # 验证能够正常处理包含换行符的文本
        self.assertEqual(len(embedding), 1)
        self.assertEqual(len(embedding[0]), 1024)
        
        # 测试空文本处理
        empty_texts = ["", "   ", "\n\n"]
        embeddings = llm_instance.embed_documents(empty_texts)
        
        # 验证空文本也能正常处理
        self.assertEqual(len(embeddings), 3)
        for emb in embeddings:
            self.assertEqual(len(emb), 1024)
        
        print("✓ 文本预处理功能测试通过")
        print("✓ 换行符处理功能正常")
        print("✓ 空文本处理功能正常")
    
    def test_unsupported_methods(self):
        """测试不支持的方法"""
        print("\n=== 测试不支持的方法 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
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
        
        # 测试不支持的重排功能
        with self.assertRaises(NotImplementedError) as cm:
            llm_instance._rerank("query", [])
        self.assertIn("do not support reranking", str(cm.exception))
        
        print("✓ 不支持的聊天功能测试通过")
        print("✓ 不支持的流式聊天功能测试通过")
        print("✓ 不支持的重排功能测试通过")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试极长文本处理
        very_long_text = "测试文本 " * 10000  # 创建一个很长的文本
        try:
            embedding = llm_instance.embed_query(very_long_text)
            self.assertEqual(len(embedding), 1024)
            print("✓ 极长文本处理正常")
        except Exception as e:
            print(f"✓ 极长文本处理异常被捕获: {type(e).__name__}")
        
        # 测试特殊字符文本
        special_text = "🔥💯✨🎉🚀💡🌟⚡️🎯🔧"
        try:
            embedding = llm_instance.embed_query(special_text)
            self.assertEqual(len(embedding), 1024)
            print("✓ 特殊字符文本处理正常")
        except Exception as e:
            print(f"✓ 特殊字符文本处理异常被捕获: {type(e).__name__}")
    
    def test_import_error_handling(self):
        """测试导入错误处理"""
        print("\n=== 测试导入错误处理 ===")
        
        if self.has_sentence_transformers:
            print("✓ sentence-transformers 已安装，跳过导入错误测试")
            return
        
        # 如果sentence-transformers未安装，测试错误处理
        config = HuggingFaceEmbedConfig(model_name="test-model")
        
        with self.assertRaises(ImportError) as cm:
            config.build()
        self.assertIn("sentence-transformers required", str(cm.exception))
        
        print("✓ 导入错误处理测试通过")
    
    def test_singleton_register(self):
        """测试注册器的单例模式"""
        print("\n=== 测试注册器单例模式 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 创建两个注册器实例
        register1 = Register()
        register2 = Register()
        
        # 验证是同一个实例
        self.assertIs(register1, register2)
        
        # 在一个实例中注册
        register1.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        
        # 在另一个实例中应该能获取到
        llm_instance = register2.get_object(self.app_name)
        self.assertIsInstance(llm_instance, HuggingFaceEmbed)
        
        print("✓ 注册器单例模式测试通过")
    
    def test_config_validation(self):
        """测试配置验证"""
        print("\n=== 测试配置验证 ===")
        
        # 测试默认配置
        config = HuggingFaceEmbedConfig(model_name="test-model")
        self.assertEqual(config.type, "huggingface_embedding")
        self.assertEqual(config.task_types, "embedding")
        self.assertEqual(config.device, "cpu")
        self.assertIsNone(config.cache_folder)
        self.assertIsNone(config.model_kwargs)
        self.assertIsNone(config.encode_kwargs)
        
        # 测试自定义配置
        custom_model_kwargs = {"trust_remote_code": True}
        custom_encode_kwargs = {"normalize_embeddings": True}
        
        config = HuggingFaceEmbedConfig(
            model_name="custom-model",
            device="cuda",
            cache_folder="/tmp/cache",
            model_kwargs=custom_model_kwargs,
            encode_kwargs=custom_encode_kwargs
        )
        
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.cache_folder, "/tmp/cache")
        self.assertEqual(config.model_kwargs, custom_model_kwargs)
        self.assertEqual(config.encode_kwargs, custom_encode_kwargs)
        
        print("✓ 默认配置验证测试通过")
        print("✓ 自定义配置验证测试通过")
    
    
    def test_build_method(self):
        """测试配置的build方法"""
        print("\n=== 测试配置的build方法 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 创建配置并构建
        config = HuggingFaceEmbedConfig(
            model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
            cache_folder=self.temp_cache_dir
        )
        embed_instance = config.build()
        
        # 验证返回的实例类型
        self.assertIsInstance(embed_instance, HuggingFaceEmbed)
        self.assertEqual(embed_instance.config, config)
        
        # 验证实例可以正常工作
        embedding = embed_instance.embed_query("测试build方法")
        self.assertEqual(len(embedding), 1024)
        
        print("✓ 配置build方法测试通过")
        print("✓ build构建的实例功能正常")
    
    def test_batch_embedding(self):
        """测试批量嵌入功能"""
        print("\n=== 测试批量嵌入功能 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试不同大小的批量
        batch_sizes = [1, 5, 10, 50]
        
        for batch_size in batch_sizes:
            texts = [f"测试文本{i}" for i in range(batch_size)]
            embeddings = llm_instance.embed_documents(texts)
            
            self.assertEqual(len(embeddings), batch_size)
            for emb in embeddings:
                self.assertEqual(len(emb), 1024)
                self.assertIsInstance(emb[0], float)
            
            print(f"✓ 批量大小 {batch_size} 测试通过")
        
        print("✓ 批量嵌入功能测试通过")
    
    def test_consistency(self):
        """测试嵌入一致性"""
        print("\n=== 测试嵌入一致性 ===")
        
        if not self.has_sentence_transformers:
            self.skipTest("sentence-transformers not installed")
        
        # 注册嵌入模型
        self.register.register(self.config_path, self.app_name, HuggingFaceEmbedConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试同一文本的多次嵌入应该一致
        text = "一致性测试文本"
        embedding1 = llm_instance.embed_query(text)
        embedding2 = llm_instance.embed_query(text)
        
        # 验证两次嵌入结果一致
        self.assertEqual(len(embedding1), len(embedding2))
        for i in range(len(embedding1)):
            self.assertAlmostEqual(embedding1[i], embedding2[i], places=6)
        
        # 测试相似文本的嵌入应该相似
        similar_texts = ["蒸汽管道", "蒸汽管道系统"]
        embeddings = llm_instance.embed_documents(similar_texts)
        
        # 计算余弦相似度
        import math
        
        def cosine_similarity(vec1, vec2):
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1))
            norm2 = math.sqrt(sum(b * b for b in vec2))
            return dot_product / (norm1 * norm2)
        
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        self.assertGreater(similarity, 0.9)  # 相似文本的相似度应该很高
        
        print("✓ 嵌入一致性测试通过")
        print(f"✓ 相似文本余弦相似度: {similarity:.6f}")


def run_tests():
    """运行所有测试"""
    print("开始运行HuggingFace嵌入模型注册器测试...")
    print("=" * 60)
    
    # 检查依赖
    try:
        import sentence_transformers
        print("✓ sentence-transformers 已安装")
    except ImportError:
        print("⚠ sentence-transformers 未安装，部分测试将被跳过")
        print("安装命令: pip install sentence-transformers")
    
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestHuggingFaceWithRegister)
    
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