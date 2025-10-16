"""
Test for complete RAG pipeline: Query Rewrite → Retrieval → Rerank → LLM Generate Answer
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.database.vector_db.faiss_config import FaissVectorDBConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
from config.encapsulation.llm.rerank.qwen import QwenRerankConfig
from config.core.query_rewrite_config import LLMQueryRewriterConfig
from config.core.retrieval.dense_config import DenseRetrieverConfig
from config.core.rerank_config import LLMRerankerConfig


def main():
    print("Testing RAG Pipeline: Query Rewrite → Retrieval → Rerank → LLM Generate Answer")

    # ===================== QUERY REWRITE COMPONENT =====================
    print("\n=== Step 1: Query Rewrite ===")

    # Create query rewriter configuration (reads from environment variables)
    llm_config = OpenAIChatConfig()
    rewriter_config = LLMQueryRewriterConfig(
        chat_llm_config=llm_config
    )

    # Build the query rewriter
    query_rewriter = rewriter_config.build()
    print(f"Query rewriter info: {query_rewriter.get_rewriter_info()}")

    # Test queries
    test_queries = [
        "Considering the information from a Wall Street Journal article and a New York Times piece on Andrew Beaton, which NFL team, known for its distinct helmet design, was the subject of Beaton's analysis regarding their strategic decisions in the off-season as reported by the Wall Street Journal, and also mentioned in the New York Times for their unexpected player trade move?",
        "Which individual, discussed across articles from Fortune, The Verge, and TechCrunch, is at the center of a legal case where the jury must assess contrasting narratives of his actions, including allegations of using a front for secret access to customer funds and committing fraud for personal gain, while also claiming to have been overwhelmed by the growth of his business ventures?",
        "Is the reporting on Taylor Swift and Travis Kelce's relationship by 'The Independent - Life and Style' inconsistent between the article published at '2023-12-06T13:55:17+00:00' stating Taylor Swift is open about her relationship with Travis Kelce and the subsequent article at '2023-12-06T14:23:01+00:00' revealing that Taylor Swift connected with Travis Kelce in July after his attempt to give her a friendship bracelet?",
        "Does the TechCrunch coverage of Sam Bankman-Fried's legal situation agree on the number of fraud and conspiracy charges he is facing, or is there a discrepancy between the articles?"
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
        embedding_config = QwenEmbeddingConfig(cache_folder="./models/Qwen", use_china_mirror=True)
        faiss_config = FaissVectorDBConfig(index_path="/home/dataarc/chenmingzhen/RAG-ARC/data/unified_faiss_index",embedding_config=embedding_config)
        dense_config = DenseRetrieverConfig(
            index_config=faiss_config
        )

        # Build the dense retriever
        dense_retriever = dense_config.build()

        # Test retrieval for each rewritten query
        all_retrieved_chunks = []
        for i, query in enumerate(rewritten_queries):
            try:
                print(f"\nRetrieving for query {i+1}: '{query}'")
                retrieved_chunks = dense_retriever.invoke(query, k=3)
                print(f"Retrieved {len(retrieved_chunks)} chunks")

                for j, chunk in enumerate(retrieved_chunks):
                    print(f"  {j+1}. ID: {chunk.id}")
                    print(f"     Content: {chunk.content[:100]}...")
                    if 'score' in chunk.metadata:
                        print(f"     Score: {chunk.metadata['score']:.4f}")

                all_retrieved_chunks.append(retrieved_chunks)
                print("-" * 50)

            except Exception as e:
                print(f"Failed to retrieve for '{query}': {e}")
                all_retrieved_chunks.append([])

        print(f"\nRetrieval completed. Retrieved chunks for {len(all_retrieved_chunks)} queries.")

    except Exception as e:
        print(f"Failed to initialize dense retriever: {e}")
        print("Make sure the FAISS index exists at ./data/unified_faiss_index")
        all_retrieved_chunks = [[] for _ in rewritten_queries]

    # ===================== RERANK COMPONENT =====================
    print("\n=== Step 3: Qwen3 Reranking ===")

    try:
        # Create reranker configuration
        qwen_llm_config = QwenRerankConfig()
        reranker_config = LLMRerankerConfig(
            rerank_llm_config=qwen_llm_config
        )

        # Build the reranker
        reranker = reranker_config.build()
        print(f"Reranker info: {reranker.get_reranker_info()}")

        # Test reranking for each query and its retrieved chunks
        all_reranked_chunks = []
        for i, (query, chunks) in enumerate(zip(rewritten_queries, all_retrieved_chunks)):
            try:
                if not chunks:
                    print(f"\nNo chunks to rerank for query {i+1}: '{query}'")
                    all_reranked_chunks.append([])
                    continue

                print(f"\nReranking for query {i+1}: '{query}'")
                print(f"Input chunks: {len(chunks)}")

                # Rerank with top_k=3
                reranked_chunks = reranker.rerank(query, chunks, top_k=3)
                print(f"Reranked {len(reranked_chunks)} chunks")

                for j, chunk in enumerate(reranked_chunks):
                    print(f"  {j+1}. ID: {chunk.id}")
                    print(f"     Content: {chunk.content[:100]}...")
                    rerank_score = chunk.metadata.get("rerank_score", "N/A")
                    original_score = chunk.metadata.get("score", "N/A")
                    print(f"     Rerank Score: {rerank_score}")
                    print(f"     Original Score: {original_score}")

                all_reranked_chunks.append(reranked_chunks)
                print("-" * 50)

            except Exception as e:
                print(f"Failed to rerank for '{query}': {e}")
                all_reranked_chunks.append(chunks)  # Fallback to original chunks

        print(f"\nReranking completed. Reranked chunks for {len(all_reranked_chunks)} queries.")

    except Exception as e:
        print(f"Failed to initialize reranker: {e}")
        print("Make sure the Qwen model is available at the specified path")
        all_reranked_chunks = all_retrieved_chunks  # Fallback to retrieved chunks

    # ===================== FINAL RESULTS SUMMARY =====================
    print("\n=== Final Results: Reranked Chunks Content ===")

    for i, (query, reranked_chunks) in enumerate(zip(rewritten_queries, all_reranked_chunks)):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: '{query}'")
        print(f"{'='*60}")

        if not reranked_chunks:
            print("No chunks found for this query.")
            continue

        for j, chunk in enumerate(reranked_chunks):
            print(f"\n--- Chunk {j+1} ---")
            print(f"ID: {chunk.id}")
            print(f"Content: {chunk.content}")
            print(f"Metadata: {chunk.metadata}")
            print("-" * 40)


if __name__ == "__main__":
    main()