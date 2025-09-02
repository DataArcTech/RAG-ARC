import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.utils.data_model import Document
from core.retrieval.bm25 import BM25Retriever
from core.retrieval.bm25 import jieba_preprocessing_func
from encapsulation.database.bm25_indexer import BM25IndexBuilder

# 加载测试数据
def load_test_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, "..", "tcl_gb_chunk.json")
    
    print("Loading documents...")
    with open(json_file_path, "r") as f:
        docs = json.load(f)
        docs_list = []
        for doc in docs:
            source = doc["metadata"]["file_name"]
            metadata = {
                "source": source
            }
            docs_list.append(Document(content=doc["content"], metadata=metadata))

    print(f"Loaded {len(docs_list)} documents")

    return docs_list



# 测试core/retrieval/bm25.py
def test_bm25_retriever():
    docs_list = load_test_data()
    
    # 创建 BM25Retriever 实例
    docs_content_list = [doc.content for doc in docs_list]
    docs_metadata_list = [doc.metadata for doc in docs_list]
    docs_ids_list = [doc.id for doc in docs_list]
    retriever = BM25Retriever.from_texts(
        texts=docs_content_list, 
        metadatas=docs_metadata_list,
        ids=docs_ids_list,
        preprocess_func=jieba_preprocessing_func, 
        k=3
    )
    
    # 测试检索功能
    query = "冷凝管"
    results = retriever._get_relevant_documents(query)
    print("Results for query '冷凝管':")
    for doc in results:
        print(doc.content)
    print('\n')
    
    # 测试分数获取
    scores = retriever.get_scores("冷凝管")
    print("Scores for query '冷凝管':", scores)

    # 测试 top-k 带分数
    top_k_with_scores = retriever.get_top_k_with_scores("冷凝管", k=2)
    print("Top-2 docs with scores for '冷凝管':")
    for doc, score in top_k_with_scores:
        print(doc.content, score)
    
    # 测试文档添加
    new_doc = Document(content="A new cat arrived.", metadata={}, id="new1")
    retriever.add_documents([new_doc])
    print("Document count after adding:", retriever.get_document_count())
    
    # 测试文档删除
    retriever.delete_documents(ids=["new1"])
    print("Document count after deleting:", retriever.get_document_count())
    
    # 测试信息获取
    info = retriever.get_bm25_info()
    print("BM25 info:", info)
    print("\n" + "="*50 + "\n")


# 测试encapsulation/database/bm25_indexer.py
def test_tantivy_bm25_retriever():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("script_dir: ", script_dir)
    
    docs_list = load_test_data()

    index_path = os.path.join(script_dir, "test_bm25_index")
    with BM25IndexBuilder(index_path=index_path, max_workers=2) as builder:
        builder.build_index(docs_list)
        builder.load_local(index_path)
        
        retriever = builder.as_retriever(top_k=3)
        
        print(f"Created retriever: {retriever}\n")
        
        # --- Normal search ---
        print("--- Normal Search for '冷凝器片距的选择依据是什么？' ---")
        results = retriever.invoke("冷凝器片距的选择依据是什么？")
        for doc in results:
            print(f"ID: {doc.id}, Score: {doc.metadata['score']:.4f}, Content: {doc.content[:100]}...")  # Truncate display

        print("\n" + "="*50 + "\n")

        # --- Search with filters ---
        print("--- Filtered Search with source filter ---")
        filtered_results = retriever.invoke(
            "冷凝器片距的选择依据是什么？", 
            filters={"source": ["Q$TD-03.085-2024 蒸发器设计规范（AC版）"]}  # 👈 Use list wrapper
        )
        for doc in filtered_results:
            print(f"ID: {doc.id}, Score: {doc.metadata['score']:.4f}, Source: {doc.metadata.get('source', 'N/A')}, Content: {doc.content[:100]}...")

        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    test_bm25_retriever()
    test_tantivy_bm25_retriever()
    print("All tests passed.")