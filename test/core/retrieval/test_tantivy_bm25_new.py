import json
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from core.utils.data_model import Document
from encapsulation.database.bm25_indexer import BM25IndexBuilder


print("Loading documents...")
with open("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-ARC/test/tcl_gb_chunk.json", "r") as f:
    docs = json.load(f)
    docs_list = []
    for doc in docs:
        source = doc["metadata"]["file_name"]
        metadata = {
            "source": source
        }
        docs_list.append(Document(content=doc["content"], metadata=metadata))

print(f"Loaded {len(docs_list)} documents")


index_path = "./my_bm25_index"
with BM25IndexBuilder(index_path=index_path, max_workers=2) as builder:
    builder.build_index(docs_list)
    builder.load_local(index_path)
    
    retriever = builder.as_retriever(top_k=3)
    
    print(f"Created retriever: {retriever}\n")
    
    # --- 普通搜索 ---
    print("--- Normal Search for '冷凝器片距的选择依据是什么？' ---")
    results = retriever.invoke("冷凝器片距的选择依据是什么？")
    for doc in results:
        print(f"ID: {doc.id}, Score: {doc.metadata['score']:.4f}, Content: {doc.content[:100]}...")  # 截断显示

    print("\n" + "="*50 + "\n")

    # --- 带过滤的搜索 ---
    print("--- Filtered Search with source filter ---")
    filtered_results = retriever.invoke(
        "冷凝器片距的选择依据是什么？", 
        filters={"source": ["Q$TD-03.085-2024 蒸发器设计规范（AC版）"]}  # 👈 使用列表包裹
    )
    for doc in filtered_results:
        print(f"ID: {doc.id}, Score: {doc.metadata['score']:.4f}, Source: {doc.metadata.get('source', 'N/A')}, Content: {doc.content[:100]}...")

    print("\n" + "="*50 + "\n")
