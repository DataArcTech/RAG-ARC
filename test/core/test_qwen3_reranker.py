"""
Test for Qwen Reranker functionality
"""

import sys
import os

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.core.reranker_config import Qwen3RerankerConfig
from config.encapsulation.llm.qwen_rerank import QwenRerankConfig
from encapsulation.data_model.schema import Document


def create_test_documents():
    """Create sample documents for testing"""
    return [
        Document(
            content="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"source": "doc1", "title": "Python Programming"},
            id="1"
        ),
        Document(
            content="Machine learning is a subset of artificial intelligence that enables computers to learn.",
            metadata={"source": "doc2", "title": "Machine Learning Basics"},
            id="2"
        ),
        Document(
            content="Climate change refers to long-term shifts in global temperatures and weather patterns.",
            metadata={"source": "doc3", "title": "Climate Science"},
            id="3"
        ),
        Document(
            content="Deep learning uses neural networks with multiple layers to model complex patterns.",
            metadata={"source": "doc4", "title": "Deep Learning"},
            id="4"
        ),
        Document(
            content="Natural language processing enables computers to understand and process human language.",
            metadata={"source": "doc5", "title": "NLP"},
            id="5"
        )
    ]


def main():
    print("Testing Qwen Reranker...")

    # Create configurations
    llm_config = QwenRerankConfig()
    reranker_config = Qwen3RerankerConfig(
        qwen3_llm_config=llm_config
    )

    # Build the reranker
    reranker = reranker_config.build()

    print(f"Reranker info: {reranker.get_reranker_info()}")

    # Create test documents
    documents = create_test_documents()
    print(f"\nTotal documents: {len(documents)}")

    # Test basic document reranking
    print("\n--- Basic Document Reranking Test ---")
    test_queries = [
        "What is Python programming?",
        "How does machine learning work?",
        "Tell me about artificial intelligence",
        "Climate change effects"
    ]

    for query in test_queries:
        try:
            reranked_docs = reranker.rerank(query, documents, top_k=3)
            print(f"Query: '{query}'")
            print(f"Returned {len(reranked_docs)} documents (top_k=3)")

            for i, doc in enumerate(reranked_docs):
                score = doc.metadata.get("rerank_score", "N/A")
                title = doc.metadata.get("title", "No title")
                print(f"  {i+1}. {title} (score: {score:.4f})")
            print("-" * 60)
        except Exception as e:
            print(f"Failed to rerank for '{query}': {e}")

    # Test error handling
    print("\n--- Error Handling Test ---")

    # Empty query test
    try:
        empty_result = reranker.rerank("", documents)
        print(f"Empty query result: {len(empty_result)} documents")
    except ValueError as e:
        print(f"Expected error for empty query: {e}")
    except Exception as e:
        print(f"Unexpected error for empty query: {e}")

    # Whitespace-only query test
    try:
        whitespace_result = reranker.rerank("   ", documents)
        print(f"Whitespace query result: {len(whitespace_result)} documents")
    except ValueError as e:
        print(f"Expected error for whitespace query: {e}")
    except Exception as e:
        print(f"Unexpected error for whitespace query: {e}")

    # Empty documents test
    try:
        empty_docs_result = reranker.rerank("test query", [])
        print(f"Empty documents result: {len(empty_docs_result)} documents")
    except Exception as e:
        print(f"Error with empty documents: {e}")

    # Test with single document
    print("\n--- Single Document Test ---")
    single_doc = [documents[0]]

    try:
        single_result = reranker.rerank("Python programming", single_doc)
        print(f"Single document test: {len(single_result)} documents returned")
        if single_result:
            score = single_result[0].metadata.get("rerank_score", "N/A")
            print(f"Document score: {score}")
    except Exception as e:
        print(f"Single document test failed: {e}")

    # Test configuration without optional parameters
    print("\n--- Minimal Configuration Test ---")
    minimal_llm_config = QwenRerankConfig()
    minimal_config = Qwen3RerankerConfig(
        qwen3_llm_config=minimal_llm_config
    )
    minimal_reranker = minimal_config.build()

    try:
        minimal_result = minimal_reranker.rerank("test query", documents[:3], top_k=2)
        print(f"Minimal config result: {len(minimal_result)} documents")
        print(f"Minimal config info: {minimal_reranker.get_reranker_info()}")
    except Exception as e:
        print(f"Minimal configuration test failed: {e}")

    # Test metadata preservation
    print("\n--- Metadata Preservation Test ---")
    test_doc = Document(
        content="Test content for metadata preservation",
        metadata={"original_score": 0.95, "source": "test", "custom_field": "preserved"},
        id="test_id"
    )

    try:
        result = reranker.rerank("test", [test_doc], top_k=1)
        if result:
            preserved_metadata = result[0].metadata
            print(f"Original metadata preserved: {preserved_metadata}")
            print(f"Rerank score added: {'rerank_score' in preserved_metadata}")
            print(f"Rerank method added: {'rerank_method' in preserved_metadata}")
    except Exception as e:
        print(f"Metadata preservation test failed: {e}")


if __name__ == "__main__":
    main()