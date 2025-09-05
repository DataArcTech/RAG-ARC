"""
Simple test to understand how HuggingFace embedding works
"""

from encapsulation.llm.huggingface import HuggingFaceLLM

def main():
    print("Testing HuggingFace LLM (Embedding)...")
    
    # Create embedding instance
    embedding = HuggingFaceLLM(
        model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
        device="cuda:0"
    )
    
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

if __name__ == "__main__":
    main()