"""
Test for Listwise Reranker functionality
"""

import sys
import os
from dotenv import load_dotenv

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables from .env file
load_dotenv()

from config.core.listwise_reranker_config import ListwiseRerankerConfig
from config.encapsulation.llm.rerank.listwise import ListwiseRerankConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from encapsulation.data_model.schema import Chunk


def create_test_chunks():
    """Create sample chunks for testing"""
    return [
        Chunk(
            content="Python is a high-level programming language known for its simplicity and readability. It is widely used in web development, data science, and machine learning.",
            metadata={"source": "doc1", "title": "Python Programming"},
            id="1"
        ),
        Chunk(
            content="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses algorithms to identify patterns.",
            metadata={"source": "doc2", "title": "Machine Learning Basics"},
            id="2"
        ),
        Chunk(
            content="Climate change refers to long-term shifts in global temperatures and weather patterns. It is primarily caused by human activities such as burning fossil fuels.",
            metadata={"source": "doc3", "title": "Climate Science"},
            id="3"
        ),
        Chunk(
            content="Deep learning uses neural networks with multiple layers to model complex patterns in data. It has achieved remarkable success in image recognition and natural language processing.",
            metadata={"source": "doc4", "title": "Deep Learning"},
            id="4"
        ),
        Chunk(
            content="Natural language processing (NLP) enables computers to understand, interpret, and generate human language. It combines linguistics and machine learning techniques.",
            metadata={"source": "doc5", "title": "NLP"},
            id="5"
        ),
        Chunk(
            content="JavaScript is a programming language primarily used for web development. It enables interactive web pages and is an essential part of web applications.",
            metadata={"source": "doc6", "title": "JavaScript"},
            id="6"
        ),
        Chunk(
            content="Data science combines statistics, programming, and domain expertise to extract insights from data. Python and R are popular languages for data science.",
            metadata={"source": "doc7", "title": "Data Science"},
            id="7"
        ),
    ]


def main():
    print("=" * 80)
    print("Testing Listwise Reranker with Real LLM")
    print("=" * 80)

    # Get API credentials from environment
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key or not base_url:
        print("ERROR: OPENAI_API_KEY and OPENAI_BASE_URL must be set in .env file")
        return

    print(f"\nUsing API Base URL: {base_url}")
    print(f"API Key: {api_key[:10]}..." if api_key else "API Key: Not set")

    # Create configurations
    chat_config = OpenAIChatConfig(
        type="openai_chat",
        model_name="gpt-4o-mini",  # Using a cost-effective model
        temperature=0.8,
        max_tokens=4096,
        openai_api_key=api_key,
        openai_base_url=base_url,
        timeout=60.0,
        max_retries=3
    )

    encap_rerank_config = ListwiseRerankConfig(
        type="listwise_rerank",
        chat_llm_config=chat_config
    )

    # Use core layer config
    reranker_config = ListwiseRerankerConfig(
        type="listwise_reranker",
        rerank_llm_config=encap_rerank_config
    )

    # Build the reranker
    print("\nBuilding listwise reranker...")
    reranker = reranker_config.build()

    print(f"Reranker info: {reranker.get_reranker_info()}")

    # Create test chunks
    chunks = create_test_chunks()
    print(f"\nTotal chunks: {len(chunks)}")

    # Test basic chunk reranking
    print("\n" + "=" * 80)
    print("Test 1: Basic Chunk Reranking")
    print("=" * 80)
    
    test_queries = [
        "What is Python programming and what is it used for?",
        "How does machine learning work?",
        "Tell me about artificial intelligence and deep learning",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)
        try:
            reranked_chunks = reranker.rerank(query, chunks, top_k=3)
            print(f"Returned {len(reranked_chunks)} chunks (top_k=3)")

            for i, chunk in enumerate(reranked_chunks):
                score = chunk.metadata.get("rerank_score", "N/A")
                title = chunk.metadata.get("title", "No title")
                content_preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
                print(f"  {i+1}. [{title}] (score: {score})")
                print(f"     {content_preview}")

        except Exception as e:
            print(f"ERROR: Failed to rerank for '{query}': {e}")
            import traceback
            traceback.print_exc()

    # Test with different top_k values
    print("\n" + "=" * 80)
    print("Test 2: Different Top-K Values")
    print("=" * 80)
    
    query = "Python programming and data science"
    for top_k in [1, 3, 5]:
        print(f"\nQuery: '{query}' (top_k={top_k})")
        print("-" * 80)
        try:
            reranked_chunks = reranker.rerank(query, chunks, top_k=top_k)
            print(f"Returned {len(reranked_chunks)} chunks")
            for i, chunk in enumerate(reranked_chunks):
                score = chunk.metadata.get("rerank_score", "N/A")
                title = chunk.metadata.get("title", "No title")
                print(f"  {i+1}. {title} (score: {score})")
        except Exception as e:
            print(f"ERROR: Failed with top_k={top_k}: {e}")

    # Test error handling
    print("\n" + "=" * 80)
    print("Test 3: Error Handling")
    print("=" * 80)

    # Empty query test
    print("\n[3.1] Empty query test:")
    try:
        empty_result = reranker.rerank("", chunks)
        print(f"  Unexpected success: {len(empty_result)} chunks")
    except ValueError as e:
        print(f"   Expected error caught: {e}")
    except Exception as e:
        print(f"   Unexpected error: {e}")

    # Whitespace-only query test
    print("\n[3.2] Whitespace query test:")
    try:
        whitespace_result = reranker.rerank("   ", chunks)
        print(f"  Unexpected success: {len(whitespace_result)} chunks")
    except ValueError as e:
        print(f"   Expected error caught: {e}")
    except Exception as e:
        print(f"   Unexpected error: {e}")

    # Empty chunks test
    print("\n[3.3] Empty chunks test:")
    try:
        empty_chunks_result = reranker.rerank("test query", [])
        print(f"   Empty chunks handled: {len(empty_chunks_result)} chunks returned")
    except Exception as e:
        print(f"   Error with empty chunks: {e}")

    # Test with single chunk
    print("\n" + "=" * 80)
    print("Test 4: Single Chunk")
    print("=" * 80)
    
    single_chunk = [chunks[0]]
    print(f"\nQuery: 'Python programming'")
    print("-" * 80)
    try:
        single_result = reranker.rerank("Python programming", single_chunk)
        print(f" Single chunk test: {len(single_result)} chunks returned")
        if single_result:
            score = single_result[0].metadata.get("rerank_score", "N/A")
            title = single_result[0].metadata.get("title", "No title")
            print(f"  Chunk: {title} (score: {score})")
    except Exception as e:
        print(f" Single chunk test failed: {e}")

    # Test metadata preservation
    print("\n" + "=" * 80)
    print("Test 5: Metadata Preservation")
    print("=" * 80)
    
    test_chunk = Chunk(
        content="Test content for metadata preservation with custom fields",
        metadata={
            "original_score": 0.95,
            "source": "test_source",
            "custom_field": "preserved_value",
            "timestamp": "2024-01-01"
        },
        id="test_id"
    )

    print("\nOriginal metadata:")
    for key, value in test_chunk.metadata.items():
        print(f"  {key}: {value}")

    try:
        result = reranker.rerank("test query", [test_chunk], top_k=1)
        if result:
            preserved_metadata = result[0].metadata
            print("\nMetadata after reranking:")
            for key, value in preserved_metadata.items():
                print(f"  {key}: {value}")
            
            print("\nVerification:")
            print(f"   Original metadata preserved: {all(k in preserved_metadata for k in test_chunk.metadata.keys())}")
            print(f"   Rerank score added: {'rerank_score' in preserved_metadata}")
            print(f"   Rerank method added: {'rerank_method' in preserved_metadata}")
            print(f"   Rerank method value: {preserved_metadata.get('rerank_method')}")
    except Exception as e:
        print(f" Metadata preservation test failed: {e}")

    # Test with custom prompt template
    print("\n" + "=" * 80)
    print("Test 6: Custom Prompt Template")
    print("=" * 80)
    
    custom_prompt = """Query: {QUERY}

Documents to rank:
{DOC_STR}

Rank the top {TOPK} most relevant documents and output their IDs as a JSON array:
```json
[id1, id2, id3, ...]
```
"""

    custom_encap_config = ListwiseRerankConfig(
        type="listwise_rerank",
        chat_llm_config=chat_config,
        prompt_template=custom_prompt
    )

    custom_core_config = ListwiseRerankerConfig(
        type="listwise_reranker",
        rerank_llm_config=custom_encap_config
    )

    custom_reranker = custom_core_config.build()
    
    print(f"\nUsing custom prompt template")
    print(f"Query: 'machine learning and AI'")
    print("-" * 80)
    try:
        custom_result = custom_reranker.rerank("machine learning and AI", chunks[:5], top_k=3)
        print(f" Custom prompt test: {len(custom_result)} chunks returned")
        for i, chunk in enumerate(custom_result):
            score = chunk.metadata.get("rerank_score", "N/A")
            title = chunk.metadata.get("title", "No title")
            print(f"  {i+1}. {title} (score: {score})")
    except Exception as e:
        print(f" Custom prompt test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

