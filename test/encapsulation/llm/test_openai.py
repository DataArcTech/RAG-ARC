"""
Simple test to understand how OpenAI Chat LLM works
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from config.encapsulation.llm.chat.openai import OpenAIChatConfig
import asyncio


def main():
    print("Testing OpenAI Chat LLM...")

    # Create OpenAI Chat LLM instance using configuration injection
    config = OpenAIChatConfig()
    chat_llm = config.build()

    print(f"Model info: {chat_llm.get_model_info()}")

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
        print()  # New line after streaming
    except Exception as e:
        print(f"Streaming chat test failed: {e}")

    # Test with different parameters in messages
    print("\n--- Different Messages Test ---")
    different_messages = [
        {"role": "user", "content": "Tell me a joke about programming."}
    ]
    try:
        response = chat_llm.chat(different_messages)
        print(f"Input: {different_messages[-1]['content']}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Different messages test failed: {e}")

    # ==================== ASYNC TESTS ====================
    print("\n" + "="*50)
    print("ASYNC CHAT FUNCTIONALITY TESTS")
    print("="*50)

    async def run_async_tests():
        # Test async chat
        print("\n--- Async Chat Test ---")
        try:
            achat_response = await chat_llm.achat(messages)
            print(f"Input: {messages[-1]['content']}")
            print(f"Async Response: {achat_response}")
            print(f"Response type: {type(achat_response)}")
            print(f"Response length: {len(achat_response)} characters")
        except Exception as e:
            print(f"Async chat test failed: {e}")

        # Test async streaming chat
        print("\n--- Async Streaming Chat Test ---")
        try:
            print("Async streaming response: ", end="", flush=True)
            async for chunk in chat_llm.astream_chat(messages):
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
            print()  # New line after streaming
        except Exception as e:
            print(f"Async streaming chat test failed: {e}")

        # Test async with different messages
        print("\n--- Async Different Messages Test ---")
        async_messages = [
            {"role": "system", "content": "You are a creative writer."},
            {"role": "user", "content": "Write a short poem about technology."}
        ]
        try:
            response = await chat_llm.achat(async_messages)
            print(f"Input: {async_messages[-1]['content']}")
            print(f"Async Response: {response[:200]}...")  # Truncate for readability
        except Exception as e:
            print(f"Async different messages test failed: {e}")

    # Run async tests
    print("\nRunning async tests...")
    try:
        asyncio.run(run_async_tests())
    except Exception as e:
        print(f"Async tests failed: {e}")

    # Test model configuration
    print("\n--- Model Configuration Test ---")
    try:
        print(f"Model Name: {chat_llm.model_name}")
        print(f"Max Tokens: {chat_llm.max_tokens}")
        print(f"Temperature: {chat_llm.temperature}")
    except Exception as e:
        print(f"Configuration test failed: {e}")


if __name__ == "__main__":
    main()