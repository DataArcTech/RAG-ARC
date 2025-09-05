"""
Simple test to understand how OpenAI LLM works for chat and embedding
"""

from encapsulation.llm.openai import OpenAILLM
import os

def main():
    print("Testing OpenAI LLM (Chat)...")
    
    # Create OpenAI LLM instance for chat
    # Note: Make sure to set OPENAI_API_KEY environment variable
    chat_llm = OpenAILLM(
        model_name="gpt-4o-mini",
        task_types=['chat'],
        base_url="https://api.gptsapi.net/v1",
        api_key="xxx",
    )
    
    print(f"Model info: {chat_llm.get_model_info()}")
    print(f"Supports chat: {chat_llm.supports_task('chat')}")
    print(f"Supports embedding: {chat_llm.supports_task('embedding')}")
    
    # Create separate embedding instance
    embedding_llm = OpenAILLM(
        model_name="text-embedding-3-small",
        task_types=['embedding'],
        base_url="https://api.gptsapi.net/v1",
        api_key="xxx",
    )
    
    # Test chat functionality
    print("\n--- Chat Test ---")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    try:
        chat_response = chat_llm.chat(messages)
        print(f"Input: {messages[-1]['content']}")
        print(f"Response: {chat_response}")
        print(f"Response type: {type(chat_response)}")
        print(f"Response length: {len(chat_response)} characters")
    except Exception as e:
        print(f"Chat test failed: {e}")
    
    # Test streaming chat
    print("\n--- Streaming Chat Test ---")
    try:
        print("Streaming response: ", end="", flush=True)
        for chunk in chat_llm.stream_chat(messages):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
            else:
                print(f"\nToken stats: {chunk}")
        print()  # New line after streaming
    except Exception as e:
        print(f"Streaming chat test failed: {e}")
    
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
        doc_embeddings = embedding_llm.embed_documents(["Document 1", "Document 2"])
        print(f"Query embedding shape: {len(query_embedding)}")
        print(f"Document embeddings shape: {len(doc_embeddings)}x{len(doc_embeddings[0])}")
    except Exception as e:
        print(f"Convenience methods test failed: {e}")
    
    # Test chat methods directly
    print(f"\n--- Chat Methods Test ---")
    try:
        chat_response = chat_llm.chat(messages)
        print(f"Direct chat() works: {len(chat_response)} chars")
    except Exception as e:
        print(f"Direct chat test failed: {e}")
    
    # Test with token counting
    print(f"\n--- Token Counting Test ---")
    try:
        response_with_tokens = chat_llm._chat(messages, return_token_count=True)
        if isinstance(response_with_tokens, tuple):
            response, token_stats = response_with_tokens
            print(f"Response: {response[:100]}...")
            print(f"Token stats: {token_stats}")
        else:
            print(f"Token counting not supported in this version")
    except Exception as e:
        print(f"Token counting test failed: {e}")

if __name__ == "__main__":
    main()