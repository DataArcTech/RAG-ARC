"""
Simple test to understand how Qwen Rerank LLM works
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from config.encapsulation.llm.rerank.qwen import QwenRerankConfig
from encapsulation.data_model.schema import Chunk


def main():
    print("Testing Qwen Rerank LLM...")

    try:
        # Create Qwen Rerank LLM instance using configuration injection
        config = QwenRerankConfig(use_china_mirror=True, cache_folder="./models/Qwen")
        qwen_llm = config.build()

        print(f"Model info: {qwen_llm.get_model_info()}")

        # Test reranking functionality with Chunk objects
        print("\n--- Chunk Object Reranking Test ---")
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

        # Create Chunk objects
        chunks = [
            Chunk(content=doc, metadata={"source": f"doc_{i}", "type": "text"}, id=f"doc_{i}")
            for i, doc in enumerate(documents_text)
        ]

        try:
            # Test basic reranking with Chunk objects
            reranked_results = qwen_llm.rerank(query, chunks)
            print(f"Query: {query}")
            print(f"Number of documents: {len(chunks)}")
            print(f"Reranked results (top 5):")

            for i, (doc_idx, score) in enumerate(reranked_results[:5]):
                doc = chunks[doc_idx]
                print(f"  Rank {i+1}: Score {score:.4f} - Doc {doc.id}")
                print(f"    Content: {doc.content[:80]}...")

            # Test with top_k parameter
            print(f"\n--- Top-K Reranking Test ---")
            top_3_results = qwen_llm.rerank(query, chunks, top_k=3)
            print(f"Top 3 results:")

            for i, (doc_idx, score) in enumerate(top_3_results):
                doc = chunks[doc_idx]
                print(f"  Rank {i+1}: Score {score:.4f} - Doc {doc.id}")
                print(f"    Content: {doc.content[:60]}...")

        except Exception as e:
            print(f"Chunk reranking test failed: {e}")

        # Test edge cases
        print(f"\n--- Edge Cases Test ---")

        # Single document
        try:
            single_doc = [Chunk(content="single document", metadata={"source": "single"}, id="single")]
            single_doc_result = qwen_llm.rerank("test query", single_doc)
            print(f"Single document reranking: {single_doc_result}")
        except Exception as e:
            print(f"Single document test failed: {e}")

        # Empty query edge case
        try:
            test_docs = chunks[:2]
            empty_query_result = qwen_llm.rerank("", test_docs)
            print(f"Empty query test: {len(empty_query_result)} results")
        except Exception as e:
            print(f"Empty query test failed: {e}")

        # Performance test with larger document set
        print(f"\n--- Performance Test ---")
        large_docs_text = [f"Document {i}: This is test document number {i} with some content." for i in range(20)]
        large_docs = [
            Chunk(content=doc_text, metadata={"source": f"perf_doc_{i}"}, id=f"perf_doc_{i}")
            for i, doc_text in enumerate(large_docs_text)
        ]

        try:
            import time
            start_time = time.time()
            large_results = qwen_llm.rerank("test document", large_docs, top_k=10)
            end_time = time.time()

            print(f"Processed {len(large_docs)} documents in {end_time - start_time:.3f} seconds")
            print(f"Top result score: {large_results[0][1]:.4f}")
            print(f"Top 3 results:")
            for i, (doc_idx, score) in enumerate(large_results[:3]):
                doc = large_docs[doc_idx]
                print(f"  Rank {i+1}: Score {score:.4f} - Doc {doc.id}")

        except Exception as e:
            print(f"Performance test failed: {e}")

        # Test different query types
        print(f"\n--- Different Query Types Test ---")
        query_tests = [
            "What is artificial intelligence?",
            "How to cook pasta?",
            "Weather today",
            "Sports scores"
        ]

        for test_query in query_tests:
            try:
                results = qwen_llm.rerank(test_query, chunks[:4], top_k=2)
                print(f"Query: '{test_query}' -> Top result score: {results[0][1]:.4f}")
            except Exception as e:
                print(f"Query '{test_query}' failed: {e}")

        # Test model configuration
        print(f"\n--- Model Configuration Test ---")
        try:
            print(f"Model Name: {qwen_llm.model_name}")
            print(f"Device: {qwen_llm.device}")
            print(f"Instruction: {qwen_llm.instruction}")
        except Exception as e:
            print(f"Configuration test failed: {e}")

    except Exception as e:
        print(f"Qwen Rerank LLM initialization failed: {e}")
        print("Note: This might fail if the model path doesn't exist or CUDA is not available")


if __name__ == "__main__":
    main()