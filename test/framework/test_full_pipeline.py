"""
注册器系统集成测试

测试使用BM25索引器配置的注册器系统工作流程。
"""
import json
import os
import sys
import tempfile
import shutil
import unittest
from typing import List, Dict, Any
from copy import deepcopy

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.utils.data_model import Document
from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig, BM25IndexBuilder
from core.retrieval.tantivy_bm25 import TantivyBM25RetrieverConfig, TantivyBM25Retriever
from framework.register import Register


class TestRegisterSystem(unittest.TestCase):
    """使用BM25索引器的注册器系统集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.register = Register()
        self.register.registrations.clear()
        self.temp_dirs = []
        self.temp_files = []
        
        # 从外部文件加载配置
        config_file_path = os.path.join(os.path.dirname(__file__), "test_configs.json")
        with open(config_file_path, 'r', encoding='utf-8') as f:
            self.test_configs = json.load(f)
    
    def tearDown(self):
        """清理测试资源"""
        # 清理临时目录
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        # 清理临时文件
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def create_temp_dir(self) -> str:
        """创建临时目录并跟踪以便清理"""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_config_file(self, config_data: Dict[str, Any]) -> str:
        """创建临时配置文件并跟踪以便清理"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f, indent=2)
            config_path = f.name
        
        self.temp_files.append(config_path)
        return config_path
    
    def load_test_documents(self, limit: int = 100) -> List[Document]:
        """从外部JSON文件加载测试文档"""
        json_file_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/test/tcl_gb_chunk.json"
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for i, item in enumerate(data[:limit]): 
            doc_id = f"doc_{i+1:04d}"
            documents.append(Document(
                id=doc_id,
                content=item.get("content", ""),
                metadata=item.get("metadata", {})
            ))
        
        print(f"✅ 已从 {json_file_path} 加载 {len(documents)} 个文档")
        return documents
    
    def test_register_workflow(self):
        """测试注册器系统工作流程：配置 -> 注册 -> 获取对象 -> 基本功能"""
        print("\n=== 测试：注册器系统工作流程 ===")
        
        # 步骤1：创建配置
        temp_dir = self.create_temp_dir()
        index_path = os.path.join(temp_dir, "test_index")
        
        config_data = deepcopy(self.test_configs["bm25_indexer_config"])
        config_data["index_path"] = index_path
        
        config_file = self.create_temp_config_file(config_data)
        print(f"✅ 已创建配置文件: {config_file}")
        
        # 步骤2：通过框架注册
        self.register.register(config_file, "bm25_indexer", BM25IndexBuilderConfig)
        self.assertIn("bm25_indexer", self.register.registrations)
        print("✅ 已通过框架成功注册BM25IndexBuilder")
        
        # 步骤3：获取注册的对象
        builder = self.register.get_object("bm25_indexer")
        self.assertIsInstance(builder, BM25IndexBuilder)
        print("✅ 已从注册器获取BM25IndexBuilder实例")
        
        # 步骤4：测试基本功能
        documents = self.load_test_documents(limit=50)  # 加载50个文档用于测试
        builder.from_documents(documents)
        
        # 验证索引已创建
        stats = builder.get_index_stats()
        self.assertGreater(stats['num_docs'], 0)
        print(f"✅ 已索引 {stats['num_docs']} 个文档")
        
        # 测试检索器创建
        retriever = builder.as_retriever()
        self.assertIsNotNone(TantivyBM25Retriever)
        print("✅ 已成功创建检索器")
        
        builder.close()
    
    def run_register_test(self):
        """运行注册器系统测试"""
        print("🚀 开始注册器系统集成测试")
        print("=" * 50)
        
        try:
            self.test_register_workflow()
            print("\n✅ 注册器系统测试通过")
            return True
        except Exception as e:
            print(f"\n❌ 注册器系统测试失败: {e}")
            return False


def run_register_test():
    """运行注册器系统测试的独立函数"""
    test_suite = TestRegisterSystem()
    test_suite.setUp()
    
    try:
        success = test_suite.run_register_test()
        return success
    finally:
        test_suite.tearDown()


if __name__ == "__main__":
    # 作为独立脚本运行
    success = run_register_test()
    if success:
        print("\n🎉 注册器系统测试成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 注册器系统测试失败。请检查上面的输出。")
        sys.exit(1)