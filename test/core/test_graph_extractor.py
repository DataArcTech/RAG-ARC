from dotenv import load_dotenv
load_dotenv() 
import sys
import os
import asyncio


# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from encapsulation.data_model.schema import Chunk
from config.core.file_management.extractor.hipporag2_extractor_config import HippoRAG2ExtractorConfig
 



async def usage_example():
    
    # 1. 配置LLM
    llm_config = OpenAIChatConfig(
        model_name="gpt-4.1-mini",
    )
    
    # 2. 配置GraphExtractor
    config = HippoRAG2ExtractorConfig(
        llm_config=llm_config,
    )
    
    # 3. 创建抽取器
    extractor = config.build()
    
    chunks = [

        Chunk(
            content="""关于本书的整理情况

本书是根据北京图书馆所藏就明刊本金陵世德堂"新刻出像官板大字《西游记》"摄影的胶卷，并参考清代六种刻本校订整理的，初版于一九五五年，以后印过多次。这次重排，又用世德堂本作了复校，并用明崇祯本作了校核，改正了初版的一些疏误。

世德堂本原书我们不曾看见。据胶卷，这个本子刻于一五九二年（明万历二十年），距作者吴承恩去世时不过十来年；虽不一定是最初刻本，但在今天所见到的许多刻本中，却是最早的。

我们所参考的清代六种刻本是：

一、《西游证道书》（清初刊本）；

二、《西游真诠》（清康熙丙子〔一六九六〕原刊本）；""",
            id="engineer_doc",
            metadata={
            "title": "ASX set to drop as Wall Street's September slump deepens",
            "author": "Stan Choe",
            "source": "The Sydney Morning Herald",
            "category": "business"
            }
        )
    ]

    lan = extractor.detect_language(chunks[0].content)
    print(lan)
    
    result = await extractor(chunks)
    
    # 打印思维导图（如果存在）
    for chunk in result:
        if chunk.metadata and 'mindmap' in chunk.metadata:
            print("\n思维导图:")
            print(chunk.metadata['mindmap'])
    
    return result


if __name__ == "__main__":
    result = asyncio.run(usage_example())
    print(f"\n处理了 {len(result)} 个chunk")
