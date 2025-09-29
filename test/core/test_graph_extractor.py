import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from encapsulation.data_model.schema import Document
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig

from dotenv import load_dotenv
load_dotenv() 


def usage_example():
    
    # 1. 配置LLM
    llm_config = OpenAIChatConfig(
        model_name="gpt-4.1-mini",
    )
    
    # 2. 配置GraphExtractor
    config = GraphExtractorConfig(
        llm_config=llm_config,
        enable_cleaning=True,
        max_rounds=3
    )
    
    # 3. 创建抽取器
    extractor = config.build()
    
    documents = [

        Document(
            content="""2020-03-20__聚灿光电科技股份有限公司__300708__聚灿光电__2019年__年度报告\n 章节:['2、资产或股权收购、出售发生的关联交易', '3、共同对外投资的关联交易', '4、关联债权债务往来', '5、其他重大关联交易', '十六、重大合同及其履行情况', '（1）托管情况', '（2）承包情况', '（3）租赁情况', '2、重大担保', '3、委托他人进行现金资产管理情况', '（2）委托贷款情况', '4、其他重大合同']\n2、资产或股权收购、出售发生的关联交易 □ 适用 √ 不适用 公司报告期未发生资产或股权收购、出售的关联交易。\n3、共同对外投资的关联交易 □ 适用 √ 不适用 公司报告期未发生共同对外投资的关联交易。\n4、关联债权债务往来 □ 适用 √ 不适用 公司报告期不存在关联债权债务往来。\n5、其他重大关联交易 √ 适用 □ 不适用 本公司作为被担保方 单位: 元 <table><thead><tr><th>担保方</th><th>担保金额</th><th>担保起始日</th><th>担保到期日</th><th>担保是否已经履行完毕</th></tr></thead><tbody><tr><td>潘华荣夫妇</td><td>120,000,000.00</td><td>2019-10-10</td><td>2022-12-15</td><td>否</td></tr><tr><td>潘华荣</td><td>80,000,000.00</td><td>2019-10-28</td><td>2022-12-15</td><td>否</td></tr></tbody></table>\n十六、重大合同及其履行情况 1、托管、承包、租赁事项情况\n（1）托管情况 □ 适用 √ 不适用 公司报告期不存在托管情况。\n（2）承包情况 □ 适用 √ 不适用 公司报告期不存在承包情况。\n（3）租赁情况 √ 适用 □ 不适用 租赁情况说明 2018年12月14日,公司与中新苏州工业园区开发集团股份有限公司签署租赁协议,租赁期自2018年12月15日至2024年12月14日,共计6年,租赁面积1,086.28平,租金85元/平/月(2019年-2021年)、91.8元/平/月(2021年-2024年),每半年支付一次。 为公司带来的损益达到公司报告期利润总额10%以上的项目 □ 适用 √ 不适用 公司报告期不存在为公司带来的损益达到公司报告期利润总额10%以上的租赁项目。\n2、重大担保 □ 适用 √ 不适用 公司报告期不存在担保情况。\n3、委托他人进行现金资产管理情况 （1）委托理财情况 □ 适用 √ 不适用 公司报告期不存在委托理财。\n（2）委托贷款情况 □ 适用 √ 不适用 公司报告期不存在委托贷款。\n4、其他重大合同 √ 适用 □ 不适用 <table><thead><tr><td>合同订立<br>公司方名<br>称</td><td>合同订立对<br>方名称</td><td>合同标的</td><td>合同签订<br>日期</td><td>定价原则</td><td>交易价<br>格(万<br>元)</td><td>是否关<br>联交易</td><td>关联关<br>系</td><td>截至报告期<br>末的执行情<br>况</td><td>披露日期</td><td>披露索引</td></tr></thead><tbody><tr><td>聚灿光电</td><td>泰谷光电科</td><td>出售4寸外延</td><td>2017年11</td><td>公允价格</td><td></td><td>否</td><td>无</td><td>按照协议正</td><td>2017年11</td><td>http://www</td></tr></tbody></table> <table><tr><td></td><td>技股份有限公司</td><td>片(框架协议)</td><td>月 23 日</td><td></td><td></td><td></td><td>常履行</td><td>月 23 日</td><td>.cninfo.com.cn</td></tr><tr><td>聚灿宿迁</td><td>苏州净化工程安装有限公司、江苏苏净科技有限公司</td><td>宿迁厂房改扩建、废水、废气、纯水工程清包及相应设备采购</td><td>2017年11月29日</td><td>公允价格, 人民币</td><td>13,720</td><td>否</td><td>无</td><td>按照协议正常履行</td><td>2017年11月30日</td><td>http://www.cninfo.com.cn</td></tr><tr><td>聚灿光电</td><td>维易科精密仪器国际贸易(上海)有限公司</td><td>采购金属有机物化学气相标准沉积设备(型号: EPIK868 C4)</td><td>2017年12月05日</td><td>公允价格, 美元</td><td>6,972</td><td>否</td><td>无</td><td>按照协议正常履行</td><td>2017年12月05日</td><td>http://www.cninfo.com.cn</td></tr><tr><td>聚灿光电</td><td>南昌中微半导体设备有限公司</td><td>采购金属有机化合物化学气相沉积设备(型号: Prismo A7)</td><td>2018年01月09日</td><td>公允价格, 人民币</td><td>62,800</td><td>否</td><td>无</td><td>合同尚未履行</td><td>2018年01月09日</td><td>http://www.cninfo.com.cn</td></tr></table>""",
            id="engineer_doc",
            metadata={
            "title": "ASX set to drop as Wall Street’s September slump deepens",
            "author": "Stan Choe",
            "source": "The Sydney Morning Herald",
            "category": "business"
            }
        )
    ]
    
    result = extractor(documents)
    
    return result



    

if __name__ == "__main__":
    print(usage_example())
