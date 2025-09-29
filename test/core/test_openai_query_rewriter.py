"""
Test for OpenAI Query Rewriter functionality
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.core.query_rewriter_config import OpenAIQueryRewriterConfig

from dotenv import load_dotenv
load_dotenv()

def main():
    print("Testing OpenAI Query Rewriter...")

    # Create configurations
    llm_config = OpenAIChatConfig()
    rewriter_config = OpenAIQueryRewriterConfig(
        openai_llm_config=llm_config
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

    # Test with instruction override
    print("\n--- Instruction Override Test ---")
    custom_instruction = "Rewrite queries to focus on technical programming concepts and code terminology"
    test_query = "How does it work?"

    try:
        rewritten_with_instruction = query_rewriter.rewrite_query(
            test_query,
            instruction=custom_instruction
        )
        print(f"Original: '{test_query}'")
        print(f"Custom instruction: '{custom_instruction}'")
        print(f"Rewritten: '{rewritten_with_instruction}'")
    except Exception as e:
        print(f"Instruction override test failed: {e}")

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
    minimal_config = OpenAIQueryRewriterConfig(
        openai_llm_config=llm_config
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
    instruction_config = OpenAIQueryRewriterConfig(
        openai_llm_config=llm_config,
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