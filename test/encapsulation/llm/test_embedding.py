"""
Simple test to understand how Embedding LLMs work (OpenAI and Qwen)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig
import asyncio


def test_openai_embedding():
    print("="*50)
    print("TESTING OPENAI EMBEDDING LLM")
    print("="*50)

    # Create OpenAI Embedding LLM instance using configuration injection
    config = OpenAIEmbeddingConfig()
    embedding_llm = config.build()

    print(f"Model info: {embedding_llm.get_model_info()}")

    # Test single text embedding
    print("\n--- Single Text Embedding Test ---")
    single_text = "This is a test sentence for embedding"
    try:
        result_single = embedding_llm.embed(single_text)
        print(f"Input: {single_text}")
        print(f"Output type: {type(result_single)}")
        print(f"Output shape: {len(result_single)} dimensions")
        print(f"First 5 values: {result_single[:5]}")
    except Exception as e:
        print(f"Single embedding test failed: {e}")

    # Test multiple texts embedding
    print("\n--- Multiple Texts Embedding Test ---")
    multiple_texts = ["Hello world", "This is another test", "Third sentence"]
    try:
        result_multiple = embedding_llm.embed(multiple_texts)
        print(f"Input: {multiple_texts}")
        print(f"Output type: {type(result_multiple)}")
        print(f"Output shape: {len(result_multiple)} texts x {len(result_multiple[0])} dimensions")
        print(f"First embedding first 5 values: {result_multiple[0][:5]}")
    except Exception as e:
        print(f"Multiple embedding test failed: {e}")

    # Test convenience methods
    print(f"\n--- Convenience Methods Test ---")
    try:
        query_embedding = embedding_llm.embed_query("What is the meaning of life?")
        doc_embeddings = embedding_llm.embed_chunks(["Chunk 1", "Chunk 2"])
        print(f"Query embedding shape: {len(query_embedding)}")
        print(f"Chunk embeddings shape: {len(doc_embeddings)}x{len(doc_embeddings[0])}")
    except Exception as e:
        print(f"Convenience methods test failed: {e}")

    # Test async functionality
    print(f"\n--- Async Embedding Test ---")
    async def test_async_openai():
        try:
            # Test async single text
            async_single = await embedding_llm.aembed("Async test sentence")
            print(f"Async single embedding shape: {len(async_single)}")

            # Test async multiple texts
            async_multiple = await embedding_llm.aembed(["Async text 1", "Async text 2"])
            print(f"Async multiple embeddings shape: {len(async_multiple)}x{len(async_multiple[0])}")
        except Exception as e:
            print(f"Async embedding test failed: {e}")

    try:
        asyncio.run(test_async_openai())
    except Exception as e:
        print(f"Async test failed: {e}")


def test_huggingface_embedding():
    print("\n" + "="*50)
    print("TESTING QWEN EMBEDDING LLM")
    print("="*50)

    try:
        # Create Qwen Embedding LLM instance using configuration injection
        config = QwenEmbeddingConfig(use_china_mirror=True, cache_folder="./models/Qwen")
        embedding_llm = config.build()

        print(f"Model info: {embedding_llm.get_model_info()}")

        # Test single text
        print("\n--- Single Text Test ---")
        single_text = "This is a test sentence"
        result_single = embedding_llm.embed(single_text)
        print(f"Input: {single_text}")
        print(f"Output type: {type(result_single)}")
        print(f"Output shape: {len(result_single)} dimensions")
        print(f"First 5 values: {result_single[:5]}")

        # Test multiple texts
        print("\n--- Multiple Texts Test ---")
        multiple_texts = ["Hello world", "This is another test", "Third sentence"]
        result_multiple = embedding_llm.embed(multiple_texts)
        print(f"Input: {multiple_texts}")
        print(f"Output type: {type(result_multiple)}")
        print(f"Output shape: {len(result_multiple)} texts x {len(result_multiple[0])} dimensions")
        print(f"First embedding first 5 values: {result_multiple[0][:5]}")

        # Show the difference
        print(f"\n--- Comparison ---")
        print(f"Single text returns: {type(result_single)} - shape {len(result_single)}")
        print(f"Multiple texts returns: {type(result_multiple)} - shape {len(result_multiple)}x{len(result_multiple[0])}")

        # Test convenience methods
        print(f"\n--- Convenience Methods Test ---")
        query_embedding = embedding_llm.embed_query("What is the meaning of life?")
        doc_embeddings = embedding_llm.embed_chunks(["Chunk 1", "Chunk 2"])
        print(f"Query embedding shape: {len(query_embedding)}")
        print(f"Chunk embeddings shape: {len(doc_embeddings)}x{len(doc_embeddings[0])}")

        # Test async functionality
        print(f"\n--- Async Embedding Test ---")
        async def test_async_huggingface():
            try:
                # Test async single text embedding
                async_single_text = "This is an async test sentence"
                result_single_async = await embedding_llm.aembed(async_single_text)
                print(f"Input: {async_single_text}")
                print(f"Output type: {type(result_single_async)}")
                print(f"Output shape: {len(result_single_async)} dimensions")
                print(f"First 5 values: {result_single_async[:5]}")

                # Test async multiple texts embedding
                async_multiple_texts = ["Async hello world", "This is another async test", "Third async sentence"]
                result_multiple_async = await embedding_llm.aembed(async_multiple_texts)
                print(f"Input: {async_multiple_texts}")
                print(f"Output type: {type(result_multiple_async)}")
                print(f"Output shape: {len(result_multiple_async)} texts x {len(result_multiple_async[0])} dimensions")
                print(f"First embedding first 5 values: {result_multiple_async[0][:5]}")

                # Compare sync vs async results
                print("\n--- Sync vs Async Comparison ---")
                test_text = "Compare sync and async"
                sync_result = embedding_llm.embed(test_text)
                async_result = await embedding_llm.aembed(test_text)

                print(f"Test text: {test_text}")
                print(f"Sync result type: {type(sync_result)}, shape: {len(sync_result)}")
                print(f"Async result type: {type(async_result)}, shape: {len(async_result)}")
                print(f"Results are equal: {sync_result == async_result}")
                print(f"Sync first 3 values: {sync_result[:3]}")
                print(f"Async first 3 values: {async_result[:3]}")
            except Exception as e:
                print(f"Async embedding test failed: {e}")

        try:
            asyncio.run(test_async_huggingface())
        except Exception as e:
            print(f"Async test failed: {e}")

    except Exception as e:
        print(f"Qwen embedding test failed: {e}")
        print("Note: This might fail if the model path doesn't exist or CUDA is not available")


def main():
    print("Testing Embedding LLMs...")

    # Test OpenAI embedding
    test_openai_embedding()

    # Test Qwen embedding
    test_huggingface_embedding()

    print(f"\n" + "="*50)
    print("ALL EMBEDDING TESTS COMPLETED")
    print("="*50)


if __name__ == "__main__":
    main()