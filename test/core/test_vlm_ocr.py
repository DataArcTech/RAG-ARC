import sys
import os
import asyncio
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.core.file_management.parser.vlm_ocr import VLMOcrParserConfig
from config.encapsulation.llm.parse.vlm_ocr import VLMOcrConfig

# 配置VLM OCR（使用OpenAI）
config = VLMOcrParserConfig(
    vlm_ocr=VLMOcrConfig(
        loading_method="openai",
        model_name="gpt-4o",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),    
        openai_base_url=os.environ.get("OPENAI_BASE_URL")
    )
)


parser = config.build()


# 读取文件内容
with open("/home/dataarc/chenmingzhen/RAG-ARC-backend/RAG-ARC/test/test_pdf.pdf", "rb") as f:
    file_data = f.read()

results = asyncio.run(parser.parse_file(file_data, "document.pdf"))

print(results)