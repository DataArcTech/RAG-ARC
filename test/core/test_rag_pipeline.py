"""
Test for complete RAG pipeline: Query Rewrite → Retrieval → Rerank → LLM Generate Answer
"""

import sys
import os

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.encapsulation.llm.openai_chat import OpenAIChatConfig
from config.encapsulation.database.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.huggingface_embedding import HuggingFaceEmbeddingConfig
from config.encapsulation.llm.qwen_rerank import QwenRerankConfig
from config.core.query_rewriter_config import OpenAIQueryRewriterConfig
from config.core.retrieval.dense_config import DenseRetrieverConfig
from config.core.reranker_config import Qwen3RerankerConfig


def main():
    print("Testing RAG Pipeline: Query Rewrite → Retrieval → Rerank → LLM Generate Answer")

    # ===================== QUERY REWRITE COMPONENT =====================
    print("\n=== Step 1: Query Rewrite ===")

    # Create query rewriter configuration
    llm_config = OpenAIChatConfig()
    rewriter_config = OpenAIQueryRewriterConfig(
        openai_llm_config=llm_config
    )

    # Build the query rewriter
    query_rewriter = rewriter_config.build()
    print(f"Query rewriter info: {query_rewriter.get_rewriter_info()}")

    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Tell me about Python programming",
        "Climate change effects"
    ]

    rewritten_queries = []
    for original_query in test_queries:
        try:
            rewritten = query_rewriter.rewrite_query(original_query)
            print(f"Original: '{original_query}'")
            print(f"Rewritten: '{rewritten}'")
            rewritten_queries.append(rewritten)
            print("-" * 50)
        except Exception as e:
            print(f"Failed to rewrite '{original_query}': {e}")
            rewritten_queries.append(original_query)  # Fallback to original

    print(f"\nQuery rewrite completed. Processed {len(rewritten_queries)} queries.")

    # ===================== RETRIEVAL COMPONENT =====================
    print("\n=== Step 2: Dense Retrieval ===")

    try:
        # Create retrieval configuration following unified_dense_config.json pattern
        embedding_config = HuggingFaceEmbeddingConfig()
        faiss_config = FaissVectorDBConfig(embedding=embedding_config)
        dense_config = DenseRetrieverConfig(
            index_config=faiss_config
        )

        # Build the dense retriever
        dense_retriever = dense_config.build()
        print(f"Dense retriever info: {dense_retriever.get_retriever_info()}")

        # Test retrieval for each rewritten query
        all_retrieved_docs = []
        for i, query in enumerate(rewritten_queries):
            try:
                print(f"\nRetrieving for query {i+1}: '{query}'")
                retrieved_docs = dense_retriever.invoke(query, k=3)
                print(f"Retrieved {len(retrieved_docs)} documents")

                for j, doc in enumerate(retrieved_docs):
                    print(f"  {j+1}. ID: {doc.id}")
                    print(f"     Content: {doc.content[:100]}...")
                    if 'score' in doc.metadata:
                        print(f"     Score: {doc.metadata['score']:.4f}")

                all_retrieved_docs.append(retrieved_docs)
                print("-" * 50)

            except Exception as e:
                print(f"Failed to retrieve for '{query}': {e}")
                all_retrieved_docs.append([])

        print(f"\nRetrieval completed. Retrieved documents for {len(all_retrieved_docs)} queries.")

    except Exception as e:
        print(f"Failed to initialize dense retriever: {e}")
        print("Make sure the FAISS index exists at ./data/unified_faiss_index")
        all_retrieved_docs = [[] for _ in rewritten_queries]

    # ===================== RERANK COMPONENT =====================
    print("\n=== Step 3: Qwen3 Reranking ===")

    try:
        # Create reranker configuration
        qwen_llm_config = QwenRerankConfig()
        reranker_config = Qwen3RerankerConfig(
            qwen3_llm_config=qwen_llm_config
        )

        # Build the reranker
        reranker = reranker_config.build()
        print(f"Reranker info: {reranker.get_reranker_info()}")

        # Test reranking for each query and its retrieved documents
        all_reranked_docs = []
        for i, (query, docs) in enumerate(zip(rewritten_queries, all_retrieved_docs)):
            try:
                if not docs:
                    print(f"\nNo documents to rerank for query {i+1}: '{query}'")
                    all_reranked_docs.append([])
                    continue

                print(f"\nReranking for query {i+1}: '{query}'")
                print(f"Input documents: {len(docs)}")

                # Rerank with top_k=3
                reranked_docs = reranker.rerank(query, docs, top_k=3)
                print(f"Reranked {len(reranked_docs)} documents")

                for j, doc in enumerate(reranked_docs):
                    print(f"  {j+1}. ID: {doc.id}")
                    print(f"     Content: {doc.content[:100]}...")
                    rerank_score = doc.metadata.get("rerank_score", "N/A")
                    original_score = doc.metadata.get("score", "N/A")
                    print(f"     Rerank Score: {rerank_score}")
                    print(f"     Original Score: {original_score}")

                all_reranked_docs.append(reranked_docs)
                print("-" * 50)

            except Exception as e:
                print(f"Failed to rerank for '{query}': {e}")
                all_reranked_docs.append(docs)  # Fallback to original docs

        print(f"\nReranking completed. Reranked documents for {len(all_reranked_docs)} queries.")

    except Exception as e:
        print(f"Failed to initialize reranker: {e}")
        print("Make sure the Qwen model is available at the specified path")
        all_reranked_docs = all_retrieved_docs  # Fallback to retrieved docs

    # ===================== FINAL RESULTS SUMMARY =====================
    print("\n=== Final Results: Reranked Documents Content ===")

    for i, (query, reranked_docs) in enumerate(zip(rewritten_queries, all_reranked_docs)):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: '{query}'")
        print(f"{'='*60}")

        if not reranked_docs:
            print("No documents found for this query.")
            continue

        for j, doc in enumerate(reranked_docs):
            print(f"\n--- Document {j+1} ---")
            print(f"ID: {doc.id}")
            print(f"Content: {doc.content}")
            print(f"Metadata: {doc.metadata}")
            print("-" * 40)

if __name__ == "__main__":
    main()