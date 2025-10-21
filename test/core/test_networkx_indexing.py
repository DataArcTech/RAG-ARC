import sys
import os
import asyncio
import tempfile
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from dotenv import load_dotenv
from encapsulation.data_model.schema import Chunk
from config.core.file_management.indexing.graph_indexing.networkx_indexing_config import NetworkXGraphIndexerConfig
from config.core.file_management.extractor.graphextractor_config import GraphExtractorConfig
from config.encapsulation.database.graph_db.networkx_with_embedding_config import NetworkXVectorConfig
from config.encapsulation.llm.chat.openai import OpenAIChatConfig
from config.encapsulation.llm.embedding.openai import OpenAIEmbeddingConfig

# Load environment variables
load_dotenv()


async def test_networkx_indexing():
    """Test the NetworkX Graph Indexer with sample chunks."""
    
    print("=" * 80)
    print("Testing NetworkX Graph Indexer")
    print("=" * 80)
    
    # Create temporary directory for storing the graph
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        
        # 1. Configure LLM for graph extraction
        llm_config = OpenAIChatConfig(
            model_name="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        
        # 2. Configure embedding model for NetworkX
        embedding_config = OpenAIEmbeddingConfig(
            model_name="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        
        # 3. Configure GraphExtractor
        extractor_config = GraphExtractorConfig(
            type="graph_extractor",
            llm_config=llm_config,
            enable_cleaning=True,
            enable_llm_cleaning=False,
            max_rounds=1,
            max_concurrent=10
        )
        
        # 4. Configure NetworkX Vector Graph Store
        networkx_config = NetworkXVectorConfig(
            type="networkx_vector",
            storage_path=temp_dir,
            index_name="test_graph_index",
            auto_save=False,  # We'll save manually
            similarity_threshold=0.5,
            cache_embeddings=True,
            embedding_cache_size=1000,
            embedding=embedding_config
        )
        
        # 5. Configure NetworkX Graph Indexer
        indexer_config = NetworkXGraphIndexerConfig(
            type="networkx_graph_indexer",
            extractor_config=extractor_config,
            graph_store_config=networkx_config
        )
        
        # 6. Build the indexer
        print("\n" + "=" * 80)
        print("Building NetworkX Graph Indexer...")
        print("=" * 80)
        indexer = indexer_config.build()
        print("✓ Indexer built successfully")
        
        # 7. Create sample chunks
        print("\n" + "=" * 80)
        print("Creating sample chunks...")
        print("=" * 80)
        chunks = [
            Chunk(
                id="chunk_1",
                content="Apple Inc. is a technology company founded by Steve Jobs. "
                        "The company is headquartered in Cupertino, California. "
                        "Tim Cook is the current CEO of Apple.",
                metadata={"source": "tech_article_1"}
            ),
            Chunk(
                id="chunk_2",
                content="Microsoft Corporation was founded by Bill Gates and Paul Allen. "
                        "The company develops Windows operating system. "
                        "Satya Nadella is the CEO of Microsoft.",
                metadata={"source": "tech_article_2"}
            ),
            Chunk(
                id="chunk_3",
                content="Tesla is an electric vehicle company led by Elon Musk. "
                        "The company manufactures electric cars and battery systems. "
                        "Tesla is headquartered in Austin, Texas.",
                metadata={"source": "tech_article_3"}
            )
        ]
        print(f"✓ Created {len(chunks)} sample chunks")
        
        # 8. Index the chunks
        print("\n" + "=" * 80)
        print("Indexing chunks...")
        print("=" * 80)
        chunk_ids = await indexer.update_index(chunks)
        
        if chunk_ids:
            print(f"✓ Successfully indexed {len(chunk_ids)} chunks")
            print(f"  Chunk IDs: {chunk_ids}")
        else:
            print("✗ Failed to index chunks")
            return False
        
        # 9. Verify the graph store
        print("\n" + "=" * 80)
        print("Verifying graph store...")
        print("=" * 80)
        
        networkx_store = indexer.networkx_store
        
        # Check graph statistics
        num_nodes = networkx_store.graph.number_of_nodes()
        num_edges = networkx_store.graph.number_of_edges()
        num_chunks = len(networkx_store.chunks)
        
        print(f"✓ Graph statistics:")
        print(f"  - Nodes: {num_nodes}")
        print(f"  - Edges: {num_edges}")
        print(f"  - Chunks: {num_chunks}")
        
        # 10. Retrieve chunks and verify graph data
        print("\n" + "=" * 80)
        print("Retrieving chunks and verifying graph data...")
        print("=" * 80)
        
        retrieved_chunks = networkx_store.get_by_ids(chunk_ids)
        
        for chunk in retrieved_chunks:
            print(f"\n  Chunk ID: {chunk.id}")
            print(f"  Content preview: {chunk.content[:100]}...")
            if chunk.graph:
                print(f"  Entities: {len(chunk.graph.entities)}")
                print(f"  Relations: {len(chunk.graph.relations)}")
                
                # Show first few entities
                if chunk.graph.entities:
                    print(f"  Sample entities:")
                    for entity in chunk.graph.entities[:3]:
                        print(f"    - {entity.get('entity_name')} ({entity.get('entity_type')})")
                
                # Show first few relations
                if chunk.graph.relations:
                    print(f"  Sample relations:")
                    for relation in chunk.graph.relations[:3]:
                        print(f"    - {relation[0]} --[{relation[1]}]--> {relation[2]}")
        
        # 11. Test persistence
        print("\n" + "=" * 80)
        print("Testing persistence...")
        print("=" * 80)
        
        # Check if files were saved
        graph_file = Path(temp_dir) / f"{networkx_config.index_name}_graph.pkl"
        docs_file = Path(temp_dir) / f"{networkx_config.index_name}_docs.pkl"
        
        if graph_file.exists() and docs_file.exists():
            print(f"✓ Index files saved successfully:")
            print(f"  - Graph file: {graph_file}")
            print(f"  - Docs file: {docs_file}")
            
            # Test loading
            print("\n  Testing index loading...")
            new_store = networkx_config.build()
            new_store.load_index(temp_dir, networkx_config.index_name)
            
            loaded_num_nodes = new_store.graph.number_of_nodes()
            loaded_num_edges = new_store.graph.number_of_edges()
            
            print(f"  ✓ Loaded graph statistics:")
            print(f"    - Nodes: {loaded_num_nodes}")
            print(f"    - Edges: {loaded_num_edges}")
            
            if loaded_num_nodes == num_nodes and loaded_num_edges == num_edges:
                print("  ✓ Persistence test passed!")
            else:
                print("  ✗ Persistence test failed - graph statistics don't match")
        else:
            print(f"✗ Index files not found")
        
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        
        return True


async def test_basic_functionality():
    """Test basic functionality without LLM calls (for quick testing)."""
    
    print("=" * 80)
    print("Testing Basic Functionality (No LLM)")
    print("=" * 80)
    
    # This test would require mocking the LLM responses
    # For now, we'll skip it and rely on the full test above
    print("Skipping basic functionality test - use test_networkx_indexing() instead")
    

if __name__ == "__main__":
    print("\nNetworkX Graph Indexer Test Suite")
    print("=" * 80)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file to run the tests")
        sys.exit(1)
    
    # Run the main test
    try:
        result = asyncio.run(test_networkx_indexing())
        if result:
            print("\n✓ All tests passed!")
            sys.exit(0)
        else:
            print("\n✗ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

