"""
Simple test to understand how Qwen LLM works for reranking
"""

from framework.config import AbstractConfig
from encapsulation.llm.qwen import QwenLLM
from encapsulation.llm.document import Document
from typing import Literal

class QwenConfig(AbstractConfig):
    """Configuration for Qwen LLM"""
    type: Literal["qwen"] = "qwen"
    model_name: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B"
    device: str = "cuda:0"
    task_types: list = ["rerank"]
    
    def build(self) -> QwenLLM:
        return QwenLLM(self)

def main():
    print("Testing Qwen LLM (Reranker)...")
    
    # Create Qwen LLM instance using configuration injection
    config = QwenConfig()
    qwen_llm = config.build()
    
    print(f"Model info: {qwen_llm.get_model_info()}")
    print(f"Supports rerank: {qwen_llm.supports_task('rerank')}")
    print(f"Supports chat: {qwen_llm.supports_task('chat')}")
    print(f"Supports embedding: {qwen_llm.supports_task('embedding')}")
    
    # Test reranking functionality with Document objects
    print("\n--- Document Object Reranking Test ---")
    query = "What is machine learning?"
    documents_text = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Cooking recipes include various ingredients and cooking methods.",
        "Deep learning uses neural networks with multiple layers to learn patterns.",
        "Weather forecasting predicts atmospheric conditions for future dates.",
        "Natural language processing helps computers understand human language.",
        "Sports news covers recent games, scores, and player statistics.",
        "Supervised learning trains models using labeled data examples."
    ]
    
    # Create Document objects
    documents = [
        Document(content=doc, metadata={"source": f"doc_{i}", "type": "text"}, id=f"doc_{i}")
        for i, doc in enumerate(documents_text)
    ]
    
    try:
        # Test basic reranking with Document objects
        reranked_results = qwen_llm.rerank(query, documents)
        print(f"Query: {query}")
        print(f"Number of documents: {len(documents)}")
        print(f"Reranked results (top 5):")
        
        for i, (doc_idx, score) in enumerate(reranked_results[:5]):
            doc = documents[doc_idx]
            print(f"  Rank {i+1}: Score {score:.4f} - Doc {doc.id}")
            print(f"    Content: {doc.content[:80]}...")
        
        # Test with top_k parameter
        print(f"\n--- Top-K Reranking Test ---")
        top_3_results = qwen_llm.rerank(query, documents, top_k=3)
        print(f"Top 3 results:")
        
        for i, (doc_idx, score) in enumerate(top_3_results):
            doc = documents[doc_idx]
            print(f"  Rank {i+1}: Score {score:.4f} - Doc {doc.id}")
            print(f"    Content: {doc.content[:60]}...")
            
    except Exception as e:
        print(f"Document reranking test failed: {e}")
    
    # Test edge cases
    print(f"\n--- Edge Cases Test ---")
    
    # Single document
    try:
        single_doc = [Document(content="single document", metadata={"source": "single"}, id="single")]
        single_doc_result = qwen_llm.rerank("test query", single_doc)
        print(f"Single document reranking: {single_doc_result}")
    except Exception as e:
        print(f"Single document test failed: {e}")
    
    # Empty query edge case
    try:
        test_docs = documents[:2]
        empty_query_result = qwen_llm.rerank("", test_docs)
        print(f"Empty query test: {len(empty_query_result)} results")
    except Exception as e:
        print(f"Empty query test failed: {e}")
    
    # Test unsupported operations
    print(f"\n--- Unsupported Operations Test ---")
    
    try:
        qwen_llm.chat([{"role": "user", "content": "Hello"}])
        print("ERROR: Chat should not be supported!")
    except Exception as e:
        print(f"✓ Chat correctly not supported: {str(e)}")
    
    try:
        qwen_llm.embed("test text")
        print("ERROR: Embedding should not be supported!")
    except Exception as e:
        print(f"✓ Embedding correctly not supported: {str(e)}")
    
    # Performance test with larger document set
    print(f"\n--- Performance Test ---")
    large_docs_text = [f"Document {i}: This is test document number {i} with some content." for i in range(20)]
    large_docs = [
        Document(content=doc_text, metadata={"source": f"perf_doc_{i}"}, id=f"perf_doc_{i}")
        for i, doc_text in enumerate(large_docs_text)
    ]
    
    try:
        import time
        start_time = time.time()
        large_results = qwen_llm.rerank("test document", large_docs, top_k=10)
        end_time = time.time()
        
        print(f"Processed {len(large_docs)} documents in {end_time - start_time:.3f} seconds")
        print(f"Top result score: {large_results[0][1]:.4f}")
        
    except Exception as e:
        print(f"Performance test failed: {e}")

if __name__ == "__main__":
    main()