import unittest
import json
import os
import sys
from typing import Dict, Any, List

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from framework.register import Register
from encapsulation.llm.openai import OpenAIConfig, OpenAILLM


class TestOpenAIWithRegister(unittest.TestCase):
    """测试使用注册器的OpenAI LLM功能"""
    
    def setUp(self):
        """测试前的设置"""
        self.register = Register()
        self.config_path = os.path.join(os.path.dirname(__file__), 'openai_test_config.json')
        self.app_name = "test_openai_llm"
        
        # 确保配置文件存在
        self._create_test_config()
    
    def _create_test_config(self):
        """创建测试配置文件"""
        config_data = {
            "type": "openai",
            "model_name": "gpt-4.1-mini",
            "task_types": "chat",
            "api_key": "sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2",
            "base_url": "https://api.gptsapi.net/v1",
            "max_retries": 3,
            "timeout": 60.0,
            "default_temperature": 0.7
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
    
    def test_register_openai_llm(self):
        """测试注册OpenAI LLM"""
        print("\n=== 测试注册OpenAI LLM ===")
        
        # 注册OpenAI LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        
        # 验证注册成功
        self.assertIn(self.app_name, self.register.registrations)
        
        # 获取注册的对象
        llm_instance = self.register.get_object(self.app_name)
        
        # 验证对象类型
        self.assertIsInstance(llm_instance, OpenAILLM)
        
        # 验证配置
        self.assertEqual(llm_instance.config.model_name, "gpt-4.1-mini")
        self.assertEqual(llm_instance.config.task_types, "chat")
        
        print(f"✓ 成功注册OpenAI LLM: {llm_instance}")
        print(f"✓ 模型名称: {llm_instance.config.model_name}")
        print(f"✓ 支持的任务类型: {llm_instance.config.task_types}")
    
    def test_chat_functionality(self):
        """测试聊天功能"""
        print("\n=== 测试聊天功能 ===")
        # 注册LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试聊天
        messages = [
            {"role": "user", "content": "你好，请简单介绍一下人工智能"}
        ]
        
        # 测试基本聊天
        response = llm_instance.chat(messages)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print(f"✓ 基本聊天响应: {response[:100]}...")
        
        # 测试带token统计的聊天
        response_with_tokens = llm_instance.chat(messages, return_token_count=True)
        self.assertIsInstance(response_with_tokens, tuple)
        self.assertEqual(len(response_with_tokens), 2)
        self.assertIsInstance(response_with_tokens[0], str)
        self.assertIsInstance(response_with_tokens[1], dict)
        self.assertIn("input_tokens", response_with_tokens[1])
        self.assertIn("output_tokens", response_with_tokens[1])
        self.assertIn("total_tokens", response_with_tokens[1])
        
        print("✓ 基本聊天功能测试通过")
        print("✓ 带token统计的聊天功能测试通过")

    
    def test_stream_chat_functionality(self):
        """测试流式聊天功能"""
        print("\n=== 测试流式聊天功能 ===")
        
        # 注册LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试流式聊天
        messages = [
            {"role": "user", "content": "请用一句话描述机器学习"}
        ]
        
        full_response = ""
        token_stats_received = False
        
        for chunk in llm_instance.stream_chat(messages, return_token_count=True):
            if isinstance(chunk, str):
                full_response += chunk
            elif isinstance(chunk, dict) and "input_tokens" in chunk:
                token_stats_received = True
                self.assertIn("input_tokens", chunk)
                self.assertIn("output_tokens", chunk)
                self.assertIn("total_tokens", chunk)
        
        self.assertGreater(len(full_response), 0)
        self.assertTrue(token_stats_received)
        
        print(f"✓ 流式聊天响应: {full_response}")
        print("✓ 流式聊天功能测试通过")
    
    def test_embedding_functionality(self):
        """测试嵌入功能"""
        print("\n=== 测试嵌入功能 ===")
        
        # 创建支持嵌入的配置
        embedding_config_path = os.path.join(os.path.dirname(__file__), 'openai_embedding_test_config.json')
        embedding_config_data = {
            "type": "openai",
            "model_name": "text-embedding-ada-002",
            "task_types": "embedding",
            "api_key": os.getenv("OPENAI_API_KEY", "sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.gptsapi.net/v1"),
            "max_retries": 3,
            "timeout": 60.0,
            "default_temperature": 0.7
        }
        
        with open(embedding_config_path, 'w', encoding='utf-8') as f:
            json.dump(embedding_config_data, f, indent=2, ensure_ascii=False)
        
        try:
            # 注册支持嵌入的LLM
            embedding_app_name = "test_openai_embedding"
            self.register.register(embedding_config_path, embedding_app_name, OpenAIConfig)
            llm_instance = self.register.get_object(embedding_app_name)
            
            # 测试单个文本嵌入
            text = "这是一个测试文本"
            embedding = llm_instance.embed_query(text)
            print(len(embedding))
            
            self.assertIsInstance(embedding, list)
            self.assertGreater(len(embedding), 0)
            self.assertIsInstance(embedding[0], float)
            
            # 测试多个文本嵌入
            texts = ["文本1", "文本2", "文本3"]
            embeddings = llm_instance.embed_documents(texts)
            
            self.assertIsInstance(embeddings, list)
            self.assertEqual(len(embeddings), 3)
            for emb in embeddings:
                self.assertIsInstance(emb, list)
                self.assertGreater(len(emb), 0)
                self.assertIsInstance(emb[0], float)
            
            print("✓ 单个文本嵌入功能测试通过")
            print("✓ 多个文本嵌入功能测试通过")
            
        finally:
            # 清理嵌入测试配置文件
            if os.path.exists(embedding_config_path):
                os.remove(embedding_config_path)
    
    def test_model_info(self):
        """测试模型信息获取"""
        print("\n=== 测试模型信息获取 ===")
        
        # 注册LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 获取模型信息
        model_info = llm_instance.get_model_info()
        
        # 验证信息完整性
        self.assertIn("model_name", model_info)
        self.assertIn("task_types", model_info)
        self.assertIn("model", model_info)
        self.assertIn("provider", model_info)
        
        self.assertEqual(model_info["model_name"], "gpt-4.1-mini")
        self.assertEqual(model_info["model"], "gpt-4.1-mini")
        self.assertEqual(model_info["provider"], "openai")
        
        print("✓ 模型信息获取测试通过")
        print(f"✓ 模型信息: {model_info}")
    
    def test_input_validation(self):
        """测试输入验证"""
        print("\n=== 测试输入验证 ===")
        
        # 注册LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试有效输入
        self.assertTrue(llm_instance.validate_input("这是一个有效的输入"))
        
        # 测试无效输入
        self.assertFalse(llm_instance.validate_input(""))  # 空字符串
        self.assertFalse(llm_instance.validate_input("   "))  # 只有空格
        self.assertFalse(llm_instance.validate_input(None))  # None值
        
        print("✓ 输入验证测试通过")
    
    def test_message_formatting(self):
        """测试消息格式化"""
        print("\n=== 测试消息格式化 ===")
        
        # 注册LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试只有用户消息
        messages = llm_instance.format_messages("用户消息")
        expected = [{"role": "user", "content": "用户消息"}]
        self.assertEqual(messages, expected)
        
        # 测试用户消息和系统消息
        messages = llm_instance.format_messages("用户消息", "系统消息")
        expected = [
            {"role": "system", "content": "系统消息"},
            {"role": "user", "content": "用户消息"}
        ]
        self.assertEqual(messages, expected)
        
        print("✓ 消息格式化测试通过")
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        
        # 注册LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试不支持的任务类型
        with self.assertRaises(ValueError):
            llm_instance.embed("测试文本")  # 当前配置不支持embedding
        
        # 测试无效的消息格式
        with self.assertRaises(ValueError):
            llm_instance.chat([])  # 空消息列表
        
        with self.assertRaises(ValueError):
            llm_instance.chat([{"role": "user"}])  # 缺少content字段
        
        print("✓ 错误处理测试通过")
    
    def test_singleton_register(self):
        """测试注册器的单例模式"""
        print("\n=== 测试注册器单例模式 ===")
        
        # 创建两个注册器实例
        register1 = Register()
        register2 = Register()
        
        # 验证是同一个实例
        self.assertIs(register1, register2)
        
        # 在一个实例中注册
        register1.register(self.config_path, self.app_name, OpenAIConfig)
        
        # 在另一个实例中应该能获取到
        llm_instance = register2.get_object(self.app_name)
        self.assertIsInstance(llm_instance, OpenAILLM)
        
        print("✓ 注册器单例模式测试通过")
    
    def test_available_models(self):
        """测试获取可用模型列表"""
        print("\n=== 测试获取可用模型列表 ===")
        
        # 注册LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 获取可用模型
        models = llm_instance.get_available_models()
        
        # 验证返回值
        self.assertIsInstance(models, list)
        
        print(f"✓ 获取到 {len(models)} 个可用模型")
        if len(models) > 0:
            print(f"✓ 示例模型: {models[:3]}")
    
    def test_rerank_not_implemented(self):
        """测试rerank功能未实现"""
        print("\n=== 测试rerank功能未实现 ===")
        
        # 注册LLM
        self.register.register(self.config_path, self.app_name, OpenAIConfig)
        llm_instance = self.register.get_object(self.app_name)
        
        # 测试rerank功能 - 应该先检查任务支持，然后抛出NotImplementedError
        with self.assertRaises(ValueError):  # 先检查任务支持
            llm_instance.rerank("query", [])
        
        print("✓ rerank功能未实现测试通过")


def run_tests():
    """运行所有测试"""
    print("开始运行OpenAI LLM注册器测试...")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestOpenAIWithRegister)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
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