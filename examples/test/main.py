import os
import sys
rag_arc_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_arc_path)


import json
from bm25 import BM25Retriever
from rag_arc.utils import Document

def chinese_preprocessing_func(text: str) -> str:
    import jieba
    return " ".join(jieba.cut(text))


docs = []
with open("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/examples/test/tcl_gb_chunk.json", "r") as f:
    data = json.load(f)

for item in data:
    docs.append(Document(content=item["content"], metadata=item["metadata"]))

bm25 = BM25Retriever.from_documents(docs, preprocess_func=chinese_preprocessing_func)

# print(bm25.get_top_k_with_scores("什么是TCL"))
bm25.save_to_disk("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/examples/test")


# import os
# import sys
# rag_arc_path = os.path.join(os.path.dirname(__file__), "..", "..")
# sys.path.insert(0, rag_arc_path)
# from bm25 import BM25Retriever
# from rag_arc.utils import Document

# bm25 = BM25Retriever.load_from_disk("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/examples/test/testbm25.pkl")

# print(bm25.get_top_k_with_scores("3 术语和定义 下列术语和定义适用于本标准。"))