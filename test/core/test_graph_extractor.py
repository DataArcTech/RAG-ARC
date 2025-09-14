#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphExtractor 真实LLM测试样例

这个测试展示如何使用真实的LLM配置来测试GraphExtractor。
注意：需要配置真实的LLM才能运行此测试。
"""

import asyncio
import os
import logging
from core.file_management.extractor.graphextractor import GraphExtractorConfig, GraphExtractor
from core.utils.data_model import Document
from encapsulation.llm.openai import OpenAIConfig

# 设置日志级别为DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def create_test_documents() -> list[Document]:
    """创建测试文档"""
    return [
        Document(
            id="doc1",
            content="""腾讯是中国最大的互联网科技公司之一，总部位于深圳。
            公司由马化腾、张志东等人于1998年11月创立。
            腾讯开发了多款著名产品，包括QQ、微信、王者荣耀等，在社交媒体和游戏领域占据领先地位。
            目前腾讯在人工智能、云计算等新兴技术领域持续投入。""",
            metadata={}
        )
    ]


def test_with_real_llm():
    """使用真实LLM进行测试"""
    print("GraphExtractor 真实LLM测试")
    print("=" * 40)
    
    # TODO: 替换为真实的LLM配置
    # 例如：
    llm_config = OpenAIConfig(
        model_name="gpt-4.1-mini",
        base_url="",
        api_key="sk-",
        task_types="chat"
    )
    
    
    # 创建GraphExtractor配置
    config = GraphExtractorConfig(
        llm_config=llm_config,
        max_rounds=2,
        enable_cleaning=False,
        entity_types=["公司", "人物", "产品", "技术", "地点", "时间"],
        relation_types=["创立", "开发", "位于", "包含", "属于", "使用"]
    )
    
    # 创建提取器
    extractor = GraphExtractor(
        config=config,
        llm=llm_config.build()
    )
    
    # 创建测试文档
    documents = create_test_documents()
    
    print(f"\n开始处理 {len(documents)} 个文档...")
    
    results = extractor(documents)  # 同步调用方式

    print(results)
    
    # 显示结果
    print("\n=== 提取结果 ===")
    for i, doc in enumerate(results):
        print(f"\n文档 {i+1} (ID: {doc.id}):")
        print(f"内容: {doc.content[:100]}...")
        
        entities = doc.metadata.get('entities', [])
        relations = doc.metadata.get('relations', [])
        
        print(f"\n提取的实体 ({len(entities)} 个):")
        for entity in entities:
            name = entity.get('entity_name', '')
            etype = entity.get('entity_type', '')
            attrs = entity.get('attributes', {})
            print(f"  - {name} ({etype})")
            if attrs:
                for key, value in attrs.items():
                    print(f"    {key}: {value}")
        
        print(f"\n提取的关系 ({len(relations)} 个):")
        for relation in relations:
            if isinstance(relation, list) and len(relation) >= 3:
                print(f"  - {relation[0]} --{relation[1]}--> {relation[2]}")
    
    print("\n✅ 测试完成！")


def main():
    """主函数"""
    try:
        test_with_real_llm()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()