"""
Test for LLM Query Rewriter functionality
"""

import sys
import os

# Add the project root to Python path for direct execution
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.core.query_rewrite_config import LLMQueryRewriterConfig

def main():
    print("Testing LLM Query Rewriter...")

    # Create configurations
    llm_config = OpenAIChatConfig()
    rewriter_config = LLMQueryRewriterConfig(
        chat_llm_config=llm_config
    )

    # Build the query rewriter
    query_rewriter = rewriter_config.build()

    print(f"Rewriter info: {query_rewriter.get_rewriter_info()}")

    # Test basic query rewriting
    print("\n--- Basic Query Rewriting Test ---")
    test_queries = [
        "What is AI?",
        "How does machine learning work?",
        "Tell me about Python programming",
        "climate change effects"
    ]

    for query in test_queries:
        try:
            rewritten = query_rewriter.rewrite_query(query)
            print(f"Original: '{query}'")
            print(f"Rewritten: '{rewritten}'")
            print("-" * 50)
        except Exception as e:
            print(f"Failed to rewrite '{query}': {e}")

    # Test with custom instruction in config
    print("\n--- Custom Instruction Configuration Test ---")
    custom_instruction = "Rewrite queries to focus on technical programming concepts and code terminology"

    custom_config = LLMQueryRewriterConfig(
        chat_llm_config=llm_config,
        instruction=custom_instruction
    )
    custom_rewriter = custom_config.build()

    test_query = "How does it work?"
    try:
        rewritten_with_instruction = custom_rewriter.rewrite_query(test_query)
        print(f"Original: '{test_query}'")
        print(f"Custom instruction: '{custom_instruction}'")
        print(f"Rewritten: '{rewritten_with_instruction}'")
    except Exception as e:
        print(f"Custom instruction test failed: {e}")

    # Empty query test
    try:
        empty_result = query_rewriter.rewrite_query("")
        print(f"Empty query result: '{empty_result}'")
    except ValueError as e:
        print(f"Expected error for empty query: {e}")
    except Exception as e:
        print(f"Unexpected error for empty query: {e}")

    # Whitespace-only query test
    try:
        whitespace_result = query_rewriter.rewrite_query("   ")
        print(f"Whitespace query result: '{whitespace_result}'")
    except ValueError as e:
        print(f"Expected error for whitespace query: {e}")
    except Exception as e:
        print(f"Unexpected error for whitespace query: {e}")

    # Test configuration without optional parameters
    print("\n--- Minimal Configuration Test ---")
    minimal_config = LLMQueryRewriterConfig(
        chat_llm_config=llm_config
        # No instruction specified (will use default)
    )
    minimal_rewriter = minimal_config.build()

    try:
        minimal_result = minimal_rewriter.rewrite_query("test query")
        print(f"Minimal config result: '{minimal_result}'")
        print(f"Minimal config info: {minimal_rewriter.get_rewriter_info()}")
    except Exception as e:
        print(f"Minimal configuration test failed: {e}")

    # Test core layer instruction configuration
    print("\n--- Core Layer Instruction Configuration Test ---")
    instruction_config = LLMQueryRewriterConfig(
        chat_llm_config=llm_config,
        instruction="Focus on scientific and academic terminology for research queries"
    )
    instruction_rewriter = instruction_config.build()

    try:
        instruction_result = instruction_rewriter.rewrite_query("machine learning basics")
        print(f"Config instruction result: '{instruction_result}'")
        print(f"Config with instruction info: {instruction_rewriter.get_rewriter_info()}")
    except Exception as e:
        print(f"Instruction configuration test failed: {e}")

if __name__ == "__main__":
    main()