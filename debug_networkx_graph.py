#!/usr/bin/env python3
"""
Debug NetworkX Graph Structure

This script helps debug why the NetworkX graph retrieval is returning 0 results.
"""

import sys
import os
import tempfile
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from config.encapsulation.database.networkx_with_embedding_config import NetworkXVectorConfig
from config.encapsulation.llm.huggingface_config import HuggingFaceEmbedConfig
from encapsulation.data_model.schema import Document, GraphData


def debug_networkx_graph():
    """Debug the NetworkX graph structure"""
    print("🔍 Debugging NetworkX Graph Structure")
    print("=" * 50)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="debug_networkx_")
    print(f"📁 Using temp directory: {temp_dir}")
    
    try:
        # Create NetworkX configuration
        config = NetworkXVectorConfig(
            storage_path=os.path.join(temp_dir, "debug_graph.pkl"),
            auto_save=True,
            similarity_threshold=0.7,
            cache_embeddings=True,
            embedding=HuggingFaceEmbedConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
        )
        
        # Initialize graph store
        graph_store = config.build()
        print("✓ NetworkX graph store initialized")
        
        # Create a simple test document with graph data
        doc = Document(
            id="test_doc",
            content="这是一个关于空调系统的测试文档。蒸发器是重要组件。",
            metadata={"topic": "空调系统"}
        )
        
        graph_data = GraphData(
            entities=[
                {
                    "id": "hvac_system", 
                    "entity_name": "空调系统", 
                    "entity_type": "设备",
                    "attributes": {
                        "功能": "制冷制热",
                        "类型": "分体式"
                    }
                },
                {
                    "id": "evaporator", 
                    "entity_name": "蒸发器", 
                    "entity_type": "组件",
                    "attributes": {
                        "材质": "铜管",
                        "作用": "吸收热量"
                    }
                }
            ],
            relations=[
                ["hvac_system", "evaporator", "包含"]
            ],
            metadata={"document_id": "test_doc"}
        )
        
        # Store document and graph data
        print("\n📥 Storing test document and graph data...")
        try:
            graph_store.add_document(doc)
            print("  ✓ Document added successfully")
        except Exception as e:
            print(f"  ❌ Document add failed: {e}")
            import traceback
            traceback.print_exc()

        try:
            graph_store.add_graph_data(graph_data, doc.id)
            print("  ✓ Graph data added successfully")
        except Exception as e:
            print(f"  ❌ Graph data add failed: {e}")
            import traceback
            traceback.print_exc()

        # Save the graph to ensure persistence
        print("  💾 Saving graph to disk...")
        graph_store.save_index(temp_dir)
        print("  ✓ Graph saved successfully")
        
        # Debug graph structure
        print(f"\n📊 Graph Statistics:")
        print(f"  Total nodes: {graph_store.graph.number_of_nodes()}")
        print(f"  Total edges: {graph_store.graph.number_of_edges()}")
        
        print(f"\n🔍 Node Analysis:")
        for node_id, node_data in graph_store.graph.nodes(data=True):
            print(f"  Node: {node_id}")
            print(f"    Type: {node_data.get('node_type', 'Unknown')}")
            print(f"    Entity Name: {node_data.get('entity_name', 'N/A')}")
            print(f"    Has Embedding: {'Yes' if node_data.get('embedding') else 'No'}")
            if node_data.get('embedding'):
                print(f"    Embedding Shape: {len(node_data['embedding'])}")
            print(f"    All attributes: {list(node_data.keys())}")
            print(f"    id_: {node_data.get('id_', 'NOT_FOUND')}")
            print()
        
        print(f"\n🔗 Edge Analysis:")
        for source, target, edge_data in graph_store.graph.edges(data=True):
            print(f"  Edge: {source} -> {target}")
            print(f"    Relation: {edge_data.get('relation_type', 'Unknown')}")
            print()
        
        # Test semantic search
        print(f"\n🔍 Testing Semantic Search:")
        
        # Test entity search
        print("  Testing entity search...")
        entity_results = []
        for node_id, node_data in graph_store.graph.nodes(data=True):
            if (node_data.get('node_type') == 'Entity' and 
                node_data.get('embedding')):
                
                # Count mentions
                mention_count = 0
                for edge in graph_store.graph.edges(data=True):
                    source, target, edge_data = edge
                    if (target == node_id and 
                        edge_data.get('relation_type') == 'MENTIONS'):
                        mention_count += 1
                
                entity_results.append({
                    'entity_id': node_data.get('id_'),
                    'entity_name': node_data.get('entity_name'),
                    'entity_type': node_data.get('entity_type'),
                    'mention_count': mention_count
                })
        
        print(f"    Found {len(entity_results)} entities with embeddings")
        for entity in entity_results:
            print(f"      - {entity['entity_name']} ({entity['entity_type']}) - {entity['mention_count']} mentions")
        
        # Test document search
        print("  Testing document search...")
        doc_results = []
        for node_id, node_data in graph_store.graph.nodes(data=True):
            if (node_data.get('node_type') == 'Document' and 
                node_data.get('embedding')):
                
                # Count entities
                entity_count = 0
                for edge in graph_store.graph.edges(data=True):
                    source, target, edge_data = edge
                    if (source == node_id and 
                        edge_data.get('relation_type') == 'MENTIONS'):
                        entity_count += 1
                
                doc_results.append({
                    'chunk_id': node_data.get('id_'),
                    'content': node_data.get('content', '')[:50] + "...",
                    'entity_count': entity_count
                })
        
        print(f"    Found {len(doc_results)} documents with embeddings")
        for doc_result in doc_results:
            print(f"      - {doc_result['chunk_id']}: {doc_result['content']} - {doc_result['entity_count']} entities")
        
        # Test actual retrieval
        print(f"\n🔍 Testing Graph Retrieval:")
        from config.core.retrieval.graph_retrieval_config import GraphRetrievalConfig
        from core.retrieval.graph_retrieveal.graph_retrieval import GraphRetrieval

        retrieval_config = GraphRetrievalConfig(
            graph_config=config,
            k1_chunks=10,
            k2_entities=5,
            top_k_entities=3,
            chunks_per_entity=2,
            alpha=0.7,
            beta=0.3
        )

        retrieval_system = retrieval_config.build()

        # Load the saved graph data into retrieval system
        print("  📂 Loading graph data into retrieval system...")
        retrieval_system.graph_store.load_index(temp_dir)

        # Check if retrieval system now has the data
        print(f"  Graph store instances match: {retrieval_system.graph_store is graph_store}")
        print(f"  Retrieval graph nodes: {retrieval_system.graph_store.graph.number_of_nodes()}")
        print(f"  Retrieval graph edges: {retrieval_system.graph_store.graph.number_of_edges()}")
        
        test_query = "空调系统的组件有哪些？"
        print(f"  Query: {test_query}")
        
        # Test candidate recall with detailed debugging
        print("  Testing semantic search step by step...")

        # Generate query embedding
        query_embedding = retrieval_system.embedding_model.embed_query(test_query)
        print(f"    Query embedding shape: {len(query_embedding)}")

        # Test graph query execution directly
        entity_results = retrieval_system._execute_graph_query("semantic_search_entities")
        print(f"    Direct entity query: {len(entity_results)} results")
        for result in entity_results[:3]:
            print(f"      - {result.get('entity_name', 'Unknown')} (id: {result.get('entity_id', 'Unknown')})")

        chunk_results = retrieval_system._execute_graph_query("semantic_search_chunks")
        print(f"    Direct chunk query: {len(chunk_results)} results")
        for result in chunk_results[:3]:
            print(f"      - {result.get('chunk_id', 'Unknown')} (content: {result.get('content', '')[:30]}...)")

        # Test entity search manually
        entity_candidates = retrieval_system.semantic_search_entities(query_embedding, 5)
        print(f"    Manual entity search: {len(entity_candidates)} candidates")
        for entity in entity_candidates:
            print(f"      - {entity.get('entity_name', 'Unknown')} (similarity: {entity.get('similarity', 0):.3f})")

        # Test chunk search manually
        chunk_candidates = retrieval_system.semantic_search_chunks(query_embedding, 5)
        print(f"    Manual chunk search: {len(chunk_candidates)} candidates")
        for chunk in chunk_candidates:
            print(f"      - {chunk.get('chunk_id', 'Unknown')} (similarity: {chunk.get('similarity', 0):.3f})")

        # Test full candidate recall
        candidates = retrieval_system.parallel_candidate_recall(test_query)
        print(f"  Entity candidates: {len(candidates.entity_candidates)}")
        print(f"  Chunk candidates: {len(candidates.chunk_candidates)}")

        if candidates.entity_candidates:
            print("  Entity candidates:")
            for entity in candidates.entity_candidates[:3]:
                print(f"    - {entity.get('entity_name', 'Unknown')} (similarity: {entity.get('similarity', 0):.3f})")

        if candidates.chunk_candidates:
            print("  Chunk candidates:")
            for chunk in candidates.chunk_candidates[:3]:
                print(f"    - {chunk.get('chunk_id', 'Unknown')} (similarity: {chunk.get('similarity', 0):.3f})")
        
        # Full retrieval test
        results = retrieval_system.retrieve(test_query, top_k=3)
        print(f"  Final results: {len(results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temp directory")


if __name__ == "__main__":
    success = debug_networkx_graph()
    if success:
        print("\n✅ Debug completed successfully")
    else:
        print("\n❌ Debug failed")
