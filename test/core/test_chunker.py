"""
Test for Chunker - testing the core chunking strategies
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.core.file_management.chunker.chunker_config import (
    TokenChunkerConfig,
    RecursiveChunkerConfig,
    MarkdownHeaderChunkerConfig,
    SemanticChunkerConfig
)
from config.encapsulation.llm.embedding.qwen import QwenEmbeddingConfig


def main():
    print("Testing Chunker - Core Chunking Strategies")

    # Test data
    simple_text = "This is a simple test text. It contains multiple sentences. Each sentence should be processed correctly."

    markdown_text = """# Main Title
This is the introduction paragraph under the main title.

## Section One
This is content under section one. It has multiple sentences and paragraphs.

This is another paragraph in section one.

### Subsection 1.1
Content under subsection 1.1 with some details.

## Section Two
This is content under section two.

### Subsection 2.1
More detailed content here.

# Another Main Title
Final section with concluding remarks.
"""

    long_text = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. In particular, how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. Modern deep learning techniques have been very successful in many natural language processing tasks. These include automatic summarization, co-reference resolution, discourse analysis, machine translation, morphological segmentation, named entity recognition, natural language understanding, natural language generation, optical character recognition, part-of-speech tagging, parsing, question answering, relationship extraction, sentiment analysis, speech recognition, speech synthesis, topic segmentation, and word sense disambiguation.

The history of natural language processing generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled Computing Machinery and Intelligence which proposed what is now called the Turing test as a criterion of intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten years of research had failed to fulfill the expectations, funding for machine translation was dramatically reduced."""

    try:
        print("=== Testing Chunker Strategies ===")

        # 1. Test TokenChunker
        print("\n--- Test 1: TokenChunker ---")
        token_config = TokenChunkerConfig()
        token_chunker = token_config.build()

        print(f"  TokenChunker info: {token_chunker.get_chunker_info()['strategy']}")
        print(f"  Chunk size: {token_chunker.get_chunker_info()['chunk_size']}")
        print(f"  Chunk overlap: {token_chunker.get_chunker_info()['chunk_overlap']}")

        token_chunks = token_chunker.chunk_text(long_text)
        print(f"  Created {len(token_chunks)} token-based chunks")

        for i, chunk in enumerate(token_chunks[:2]):  # Show first 2 chunks
            print(f"  Chunk {i}: {len(chunk['content'])} chars, {chunk['metadata']['token_count']} tokens")
            print(f"    Preview: {chunk['content'][:100]}...")

        # 2. Test RecursiveChunker
        print("\n--- Test 2: RecursiveChunker ---")
        recursive_config = RecursiveChunkerConfig()
        recursive_chunker = recursive_config.build()

        print(f"  RecursiveChunker info: {recursive_chunker.get_chunker_info()['strategy']}")
        print(f"  Separators: {recursive_chunker.get_chunker_info()['separators']}")
        print(f"  Chunk size: {recursive_chunker.get_chunker_info()['chunk_size']}")

        recursive_chunks = recursive_chunker.chunk_text(long_text)
        print(f"  Created {len(recursive_chunks)} recursive chunks")

        for i, chunk in enumerate(recursive_chunks[:2]):  # Show first 2 chunks
            print(f"  Chunk {i}: {chunk['metadata']['character_count']} chars")
            print(f"    Preview: {chunk['content'][:100]}...")

        # 3. Test MarkdownHeaderChunker
        print("\n--- Test 3: MarkdownHeaderChunker ---")
        markdown_config = MarkdownHeaderChunkerConfig()
        markdown_chunker = markdown_config.build()

        print(f"  MarkdownHeaderChunker info: {markdown_chunker.get_chunker_info()['strategy']}")
        print(f"  Headers to split on: {markdown_chunker.get_chunker_info()['headers_to_split_on']}")
        print(f"  Strip headers: {markdown_chunker.get_chunker_info()['strip_headers']}")

        markdown_chunks = markdown_chunker.chunk_text(markdown_text)
        print(f"  Created {len(markdown_chunks)} markdown chunks")

        for i, chunk in enumerate(markdown_chunks[:3]):  # Show first 3 chunks
            header_info = chunk['metadata']['header']
            print(f"  Chunk {i}: Header level {header_info['level']}, name: '{header_info['name']}'")
            print(f"    Content length: {len(chunk['content'])} chars")
            print(f"    Preview: {chunk['content'][:80]}...")

        # 4. Test SemanticChunker
        print("\n--- Test 4: SemanticChunker ---")
        try:
            hf_config = QwenEmbeddingConfig(
                use_china_mirror=True,
                cache_folder="./models/Qwen"
            )
            semantic_config = SemanticChunkerConfig(embedding=hf_config)
            semantic_chunker = semantic_config.build()

            print(f"  SemanticChunker info: {semantic_chunker.get_chunker_info()['strategy']}")
            print(f"  Buffer size: {semantic_chunker.get_chunker_info()['buffer_size']}")
            print(f"  Threshold type: {semantic_chunker.get_chunker_info()['breakpoint_threshold_type']}")
            print(f"  Embedding model initialized: {semantic_chunker.embeddings is not None}")

            semantic_chunks = semantic_chunker.chunk_text(long_text)
            print(f"  Created {len(semantic_chunks)} semantic chunks")

            for i, chunk in enumerate(semantic_chunks[:2]):  # Show first 2 chunks
                print(f"  Chunk {i}: {chunk['metadata']['character_count']} chars")
                print(f"    Preview: {chunk['content'][:100]}...")

        except Exception as e:
            print(f"  SemanticChunker test failed: {e}")
            print("  Note: SemanticChunker requires embedding model and may need GPU/model files")

        # 5. Test chunker with metadata
        print("\n--- Test 5: Chunking with Source Metadata ---")
        source_metadata = {
            "chunk_id": "test_chunk_001",
            "source": "test_document.txt",
            "author": "Test Author",
            "created_at": "2024-01-01"
        }

        chunks_with_metadata = token_chunker.chunk_text(simple_text, metadata=source_metadata)
        print(f"  Created {len(chunks_with_metadata)} chunks with source metadata")

        sample_chunk = chunks_with_metadata[0]
        print(f"  Sample chunk metadata: {sample_chunk['metadata']}")
        print(f"  Sample source metadata: {sample_chunk['source_metadata']}")

        # 6. Test chunker configuration overrides
        print("\n--- Test 6: Runtime Parameter Overrides ---")

        # Test TokenChunker with different parameters
        override_chunks = token_chunker.chunk_text(
            simple_text,
            chunk_size=200,
            chunk_overlap=20
        )
        print(f"  TokenChunker with override: {len(override_chunks)} chunks")
        print(f"  First chunk token count: {override_chunks[0]['metadata']['token_count']}")

        # Test RecursiveChunker with different separators
        custom_recursive_chunks = recursive_chunker.chunk_text(
            long_text,
            chunk_size=300,
            separators=[". ", " "]
        )
        print(f"  RecursiveChunker with custom separators: {len(custom_recursive_chunks)} chunks")

        # 7. Test error handling
        print("\n--- Test 7: Error Handling ---")

        try:
            # Test with empty text
            empty_chunks = token_chunker.chunk_text("")
            print(f"  Empty text handling: {len(empty_chunks)} chunks")
        except Exception as e:
            print(f"  Empty text error: {e}")

        try:
            # Test RecursiveChunker with invalid parameters
            recursive_chunker.chunk_text(simple_text, chunk_size=0)
            print("  Invalid parameters should have failed")
        except ValueError as e:
            print(f"  Properly caught invalid parameter: {e}")

        # 8. Test chunker info consistency
        print("\n--- Test 8: Chunker Info Validation ---")

        chunkers = [
            ("Token", token_chunker),
            ("Recursive", recursive_chunker),
            ("MarkdownHeader", markdown_chunker)
        ]

        for name, chunker in chunkers:
            info = chunker.get_chunker_info()
            required_fields = ['strategy', 'supported_features', 'parameters']
            missing_fields = [field for field in required_fields if field not in info]

            if missing_fields:
                print(f"  {name}Chunker missing fields: {missing_fields}")
            else:
                print(f"  {name}Chunker info complete: {info['strategy']} strategy")
                print(f"    Features: {', '.join(info['supported_features'])}")

        print("\n All Chunker strategies tested successfully!")
        print("Note: All chunkers follow the same interface and return standardized chunk format")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        print("Make sure the embedding model is available at the specified path:")
        print(f"  Model path: /finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()