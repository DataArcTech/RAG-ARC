#!/usr/bin/env python3
"""
重构后的GraphExtractor使用示例
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.file_management.extractor.graphextractor import GraphExtractor
from config.encapsulation.llm.openai_config import OpenAIConfig
from encapsulation.data_model.data_model import Document, GraphData
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig



def usage_example():
    
    # 1. 配置LLM
    llm_config = OpenAIConfig(
        model_name="gpt-4.1-mini",
        api_key="sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2",  # 测试用的假key
        base_url="https://api.gptsapi.net/v1",
    )
    
    # 2. 配置GraphExtractor
    config = GraphExtractorConfig(
        llm_config=llm_config,
        # entity_types=["Person", "Company", "Product", "Technology"],
        # relation_types=["works_at", "develops", "competes_with", "uses"],
        # entity_examples=[
        #     {"name": "张三", "type": "Person", "attributes": {"职位": "工程师", "年龄": "30"}},
        #     {"name": "苹果公司", "type": "Company", "attributes": {"行业": "科技", "成立年份": "1976"}},
        #     {"name": "iPhone", "type": "Product", "attributes": {"类型": "智能手机", "发布年份": "2007"}}
        # ],
        # relation_examples=[
        #     ["e1", "works_at", "e2"],
        #     ["e2", "develops", "e3"],
        #     ["e2", "competes_with", "e4"]
        # ],
        enable_multi_round=True,
        enable_cleaning=True,
        enable_llm_cleaning=True,
        max_rounds=3
    )
    
    # 3. 创建抽取器
    extractor = config.build()
    
    documents = [

        Document(
            content="""1.3.5 铜基体溶液(100 g/L):称取20.00 g纯铜(1.3.1)置于400 mL烧杯中,分次加入160 mL硝酸(1.3.2),冷溶。待激烈反应停止后,低温加热至完全溶解,煮沸驱除氮的氧化物,冷却至室温。移入200 mL容量瓶中,以水稀释至刻度,混匀。\n#### 1.3.6 铁标准贮存溶液:称取0.2000 g金属铁(铁的质量分数≥99.95%)置于150 mL烧杯中,加入14 mL盐酸(1.3.4),盖上表皿,低温加热至完全溶解,冷却至室温。移入500 mL容量瓶中,以水稀释至刻度,混匀。此溶液1 mL含400 µg铁。\n#### 1.3.7 铁标准溶液A:移取5.00 mL铁标准贮存溶液(1.3.6)于200 mL容量瓶中,以水稀释至刻度,混匀。此溶液1 mL含10 µg铁。\n#### 1.3.8 铁标准溶液B:移取20.00 mL铁标准溶液A(1.3.7)于200 mL容量瓶中,以水稀释至刻度,混匀。此溶液1 mL含1 µg铁。1.4 仪器 1.4.1 石墨炉原子吸收光谱仪:配备电热原子化器、微量取样器或自动进样器,铁空心阴极灯及塞曼效应背景校正装置。1.4.2 所用石墨炉原子吸收光谱仪应达到下列指标:\n——最低灵敏度:工作曲线中所用等差系列标准溶液中浓度最大者,其吸光度应不低于0.300。\n——工作曲线的相关系数不低于0.995。\n——精密度最低要求:用最高浓度的标准溶液,测量10次吸光度,计算其平均值和标准偏差。该标准偏差不应超过该吸光度平均值的1.5%。用最低浓度的标准溶液(不是浓度为零的标准溶液),测量10次吸光度,计算其标准偏差。该标准偏差不应超过最高浓度标准溶液吸光度平均值的0.5%。""",
            id="engineer_doc"
        )
    ]
    
    result = extractor(documents)
    
    return result



    

if __name__ == "__main__":
    print(usage_example())
