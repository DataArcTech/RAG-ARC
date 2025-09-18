"""
Simple test to understand how HuggingFace embedding works
"""

from framework.config import AbstractConfig
from encapsulation.llm.huggingface import HuggingFaceLLM
from typing import Literal
import asyncio

class HuggingFaceConfig(AbstractConfig):
    """Configuration for HuggingFace LLM"""
    type: Literal["huggingface"] = "huggingface"
    model_name: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B"
    device: str = "cuda:0"
    task_types: list = ["embedding"]
    
    def build(self) -> HuggingFaceLLM:
        return HuggingFaceLLM(self)

def main():
    print("Testing HuggingFace LLM (Embedding)...")
    
    # Create embedding instance using configuration injection
    config = HuggingFaceConfig()
    embedding = config.build()
    
    print(f"Model info: {embedding.get_model_info()}")
    print(f"Supports embedding: {embedding.supports_task('embedding')}")
    print(f"Supports chat: {embedding.supports_task('chat')}")
    
    # Test single text
    print("\n--- Single Text Test ---")
    single_text = "This is a test sentence"
    result_single = embedding.embed(single_text)  # Use public method instead of _embed
    print(f"Input: {single_text}")
    print(f"Output type: {type(result_single)}")
    print(f"Output shape: {len(result_single)} dimensions")
    print(f"First 5 values: {result_single[:5]}")
    
    # Test multiple texts
    print("\n--- Multiple Texts Test ---")
    multiple_texts = ["Hello world", "This is another test", "Third sentence"]
    result_multiple = embedding.embed(multiple_texts)  # Use public method instead of _embed
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
    query_embedding = embedding.embed_query("What is the meaning of life?")
    doc_embeddings = embedding.embed_documents(["Document 1", "Document 2"])
    print(f"Query embedding shape: {len(query_embedding)}")
    print(f"Document embeddings shape: {len(doc_embeddings)}x{len(doc_embeddings[0])}")

    # ==================== ASYNC EMBEDDING TESTS ====================
    print("\n" + "="*50)
    print("ASYNC EMBEDDING TESTS")
    print("="*50)

    async def run_async_embedding_tests():
        # Test async single text embedding
        print("\n--- Async Single Text Test ---")
        async_single_text = "This is an async test sentence"
        try:
            result_single_async = await embedding.aembed(async_single_text)
            print(f"Input: {async_single_text}")
            print(f"Output type: {type(result_single_async)}")
            print(f"Output shape: {len(result_single_async)} dimensions")
            print(f"First 5 values: {result_single_async[:5]}")
        except Exception as e:
            print(f"Async single text test failed: {e}")

        # Test async multiple texts embedding
        print("\n--- Async Multiple Texts Test ---")
        async_multiple_texts = ["Async hello world", "This is another async test", "Third async sentence"]
        try:
            result_multiple_async = await embedding.aembed(async_multiple_texts)
            print(f"Input: {async_multiple_texts}")
            print(f"Output type: {type(result_multiple_async)}")
            print(f"Output shape: {len(result_multiple_async)} texts x {len(result_multiple_async[0])} dimensions")
            print(f"First embedding first 5 values: {result_multiple_async[0][:5]}")
        except Exception as e:
            print(f"Async multiple texts test failed: {e}")

        # Compare sync vs async results
        print("\n--- Sync vs Async Comparison ---")
        try:
            test_text = "Compare sync and async"
            sync_result = embedding.embed(test_text)
            async_result = await embedding.aembed(test_text)

            print(f"Test text: {test_text}")
            print(f"Sync result type: {type(sync_result)}, shape: {len(sync_result)}")
            print(f"Async result type: {type(async_result)}, shape: {len(async_result)}")
            print(f"Results are equal: {sync_result == async_result}")
            print(f"Sync first 3 values: {sync_result[:3]}")
            print(f"Async first 3 values: {async_result[:3]}")
        except Exception as e:
            print(f"Sync vs async comparison failed: {e}")

    # Run async tests
    print("\nRunning async embedding tests...")
    try:
        asyncio.run(run_async_embedding_tests())
    except Exception as e:
        print(f"Async embedding tests failed: {e}")

if __name__ == "__main__":
    main()