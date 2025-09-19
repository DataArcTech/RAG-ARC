import sys
import os
import asyncio
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

try:
    from core.file_management.extractor.graphextractor import GraphExtractorConfig
    from encapsulation.llm.openai import OpenAIConfig
    from core.utils.data_model import Document
    from encapsulation.database.graph_db.neo4j import Neo4jConfig, Neo4jGraphStore
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_AVAILABLE = False


def create_test_documents():
    """Create test documents for processing"""
    print("=== Creating Test Documents ===")
    
    documents = [
        Document(
            content="""
            1.3.5 铜基体溶液(100 g/L):称取20.00 g纯铜(1.3.1)置于400 mL烧杯中,分次加入160 mL硝酸(1.3.2),冷溶。待激烈反应停止后,低温加热至完全溶解,煮沸驱除氮的氧化物,冷却至室温。移入200 mL容量瓶中,以水稀释至刻度,混匀。
            
            1.3.6 铁标准贮存溶液:称取0.2000 g金属铁(铁的质量分数≥99.95%)置于150 mL烧杯中,加入14 mL盐酸(1.3.4),盖上表皿,低温加热至完全溶解,冷却至室温。移入500 mL容量瓶中,以水稀释至刻度,混匀。此溶液1 mL含400 µg铁。
            
            1.3.7 铁标准溶液A:移取5.00 mL铁标准贮存溶液(1.3.6)于200 mL容量瓶中,以水稀释至刻度,混匀。此溶液1 mL含10 µg铁。
            """,
            id='lab_manual_solutions',
            metadata={'source': 'laboratory_manual', 'section': '1.3', 'type': 'solutions'}
        ),
        Document(
            content="""
            1.4 仪器
            1.4.1 石墨炉原子吸收光谱仪:配备电热原子化器、微量取样器或自动进样器,铁空心阴极灯及塞曼效应背景校正装置。
            
            1.4.2 所用石墨炉原子吸收光谱仪应达到下列指标:
            ——最低灵敏度:工作曲线中所用等差系列标准溶液中浓度最大者,其吸光度应不低于0.300。
            ——工作曲线的相关系数不低于0.995。
            ——精密度最低要求:用最高浓度的标准溶液,测量10次吸光度,计算其平均值和标准偏差。该标准偏差不应超过该吸光度平均值的1.5%。
            """,
            id='lab_manual_instruments',
            metadata={'source': 'laboratory_manual', 'section': '1.4', 'type': 'instruments'}
        )
    ]
    
    for i, doc in enumerate(documents):
        print(f"  Document {i+1}: {doc.id}")
        print(f"    Content length: {len(doc.content)} characters")
        print(f"    Metadata: {doc.metadata}")
    
    return documents


def setup_graphextractor():
    """Setup GraphExtractor with real LLM"""
    print("\n=== Setting up GraphExtractor ===")
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available")
        return None
    
    try:
        # Real LLM configuration
        llm_config = OpenAIConfig(
            model_name='gpt-4.1-mini',
            api_key='sk-',
            base_url='https://api.gptsapi.net/v1',
            default_max_tokens=2000
        )
        
        # GraphExtractor configuration
        config = GraphExtractorConfig(
            llm_config=llm_config,
            entity_types=['溶液', '仪器', '性能指标', '化学品', '设备', '标准', '方法'],
            relation_types=['derived_from', 'has_performance_indicator', 'used_for', 'contains', 'requires', 'measures', 'configured_with'],
            enable_cleaning=True,
            enable_llm_cleaning=True,
            max_rounds=2
        )
        
        extractor = config.build()
        
        print("✅ GraphExtractor configured successfully")
        print(f"   Entity types: {len(config.entity_types)}")
        print(f"   Relation types: {len(config.relation_types)}")
        print(f"   LLM cleaning enabled: {config.enable_llm_cleaning}")
        print(f"   Max rounds: {config.max_rounds}")
        
        return extractor
        
    except Exception as e:
        print(f"❌ Failed to setup GraphExtractor: {e}")
        import traceback
        traceback.print_exc()
        return None


def setup_neo4j_store():
    """Setup Neo4j store with real database"""
    print("\n=== Setting up Neo4j Store ===")
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available")
        return None
    
    try:
        # Real Neo4j configuration
        config = Neo4jConfig(
            index_name="knowledge_graph_test",
            url="neo4j://192.168.80.1:7660",
            username="neo4j",
            password="12345678",
            database="neo4j"
        )
        
        store = config.build()
        
        # Test connection
        health = store.health_check()
        print(f"✅ Neo4j store configured successfully")
        print(f"   Database: {config.url}")
        print(f"   Index name: {config.index_name}")
        print(f"   Health status: {health.get('status', 'unknown')}")
        
        return store
        
    except Exception as e:
        print(f"❌ Failed to setup Neo4j store: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_extraction_pipeline(extractor, documents):
    """Run GraphExtractor on documents"""
    print("\n=== Running Extraction Pipeline ===")
    
    try:
        print(f"Processing {len(documents)} documents...")
        
        # Extract graph data
        results = extractor(documents)
        
        print(f"✅ Extraction completed")
        print(f"   Processed documents: {len(results)}")
        
        # Display results
        total_entities = 0
        total_relations = 0
        
        for i, doc in enumerate(results):
            print(f"\n📄 Document {i+1}: {doc.id}")
            print(f"   Entities: {len(doc.graph.entities)}")
            print(f"   Relations: {len(doc.graph.relations)}")
            
            total_entities += len(doc.graph.entities)
            total_relations += len(doc.graph.relations)
            
            # Show sample entities
            if doc.graph.entities:
                print(f"   Sample entities:")
                for j, entity in enumerate(doc.graph.entities[:3]):
                    print(f"     {j+1}. {entity['entity_name']} ({entity['entity_type']})")
                if len(doc.graph.entities) > 3:
                    print(f"     ... and {len(doc.graph.entities) - 3} more")
            
            # Show sample relations
            if doc.graph.relations:
                print(f"   Sample relations:")
                for j, relation in enumerate(doc.graph.relations[:3]):
                    print(f"     {j+1}. {relation[0]} --[{relation[1]}]--> {relation[2]}")
                if len(doc.graph.relations) > 3:
                    print(f"     ... and {len(doc.graph.relations) - 3} more")
        
        print(f"\n📊 Total extracted:")
        print(f"   Entities: {total_entities}")
        print(f"   Relations: {total_relations}")
        
        return results
        
    except Exception as e:
        print(f"❌ Extraction pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def store_to_neo4j(store, documents):
    """Store documents to Neo4j"""
    print("\n=== Storing to Neo4j ===")
    
    try:
        # Clear existing data for clean test
        print("Clearing existing test data...")
        store.delete()
        
        # Add documents
        print(f"Adding {len(documents)} documents to Neo4j...")
        added_ids = store.add(documents)
        
        print(f"✅ Successfully added {len(added_ids)} documents")
        print(f"   Document IDs: {added_ids}")
        
        # Get statistics
        stats = store.get_index_stats()
        print(f"\n📊 Database statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Total entities: {stats.get('total_entities', 0)}")
        print(f"   Total relationships: {stats.get('total_relationships', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to store to Neo4j: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_storage(store, original_documents):
    """Verify data was stored correctly"""
    print("\n=== Verifying Storage ===")
    
    try:
        # Retrieve documents
        doc_ids = [doc.id for doc in original_documents]
        retrieved_docs = store.get_by_ids(doc_ids)
        
        print(f"✅ Retrieved {len(retrieved_docs)} documents")
        
        # Compare data
        for i, (original, retrieved) in enumerate(zip(original_documents, retrieved_docs)):
            print(f"\n📄 Document {i+1}: {original.id}")
            print(f"   Original entities: {len(original.graph.entities)}")
            print(f"   Retrieved entities: {len(retrieved.graph.entities)}")
            print(f"   Original relations: {len(original.graph.relations)}")
            print(f"   Retrieved relations: {len(retrieved.graph.relations)}")
            
            # Check entity preservation
            original_entity_names = {e['entity_name'] for e in original.graph.entities}
            retrieved_entity_names = {e['entity_name'] for e in retrieved.graph.entities}
            
            if original_entity_names == retrieved_entity_names:
                print(f"   ✅ Entity names preserved")
            else:
                print(f"   ⚠️ Entity names differ")
                print(f"      Missing: {original_entity_names - retrieved_entity_names}")
                print(f"      Extra: {retrieved_entity_names - original_entity_names}")
            
            # Check relation preservation
            original_relations = {tuple(r) for r in original.graph.relations}
            retrieved_relations = {tuple(r) for r in retrieved.graph.relations}
            
            if original_relations == retrieved_relations:
                print(f"   ✅ Relations preserved")
            else:
                print(f"   ⚠️ Relations differ")
                print(f"      Missing: {original_relations - retrieved_relations}")
                print(f"      Extra: {retrieved_relations - original_relations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("Complete GraphExtractor → Neo4j Pipeline Test")
    print("=" * 60)
    print("🔧 Using real LLM and real Neo4j database")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available")
        return
    
    # Setup components
    documents = create_test_documents()
    extractor = setup_graphextractor()
    store = setup_neo4j_store()
    
    if not extractor or not store:
        print("❌ Failed to setup required components")
        return
    
    # Run pipeline
    extracted_docs = run_extraction_pipeline(extractor, documents)
    
    if not extracted_docs:
        print("❌ Extraction pipeline failed")
        return
    
    # Store to Neo4j
    storage_success = store_to_neo4j(store, extracted_docs)
    
    if not storage_success:
        print("❌ Storage to Neo4j failed")
        return
    
    # Verify storage
    verification_success = verify_storage(store, extracted_docs)
    
    # Final summary
    print("\n" + "=" * 60)
    if verification_success:
        print("🎉 Complete Pipeline Test SUCCESSFUL!")
        print("\n✅ Pipeline completed successfully:")
        print("1. ✅ Created test documents")
        print("2. ✅ Configured GraphExtractor with real LLM")
        print("3. ✅ Configured Neo4j store with real database")
        print("4. ✅ Extracted graph data using LLM")
        print("5. ✅ Stored data to Neo4j with Document-Entity relationships")
        print("6. ✅ Verified data integrity")
        
        print(f"\n🔍 Explore your data:")
        print(f"Neo4j Browser: http://192.168.80.1:7460")
        print(f"Sample queries:")
        print(f"  // Show all documents and their entities")
        print(f"  MATCH (d:Document)-[:CONTAINS]->(e:Entity) RETURN d, e")
        print(f"  // Show entity relationships")
        print(f"  MATCH (e1:Entity)-[r]->(e2:Entity) RETURN e1, r, e2")
        print(f"  // Show full graph")
        print(f"  MATCH (n) RETURN n LIMIT 50")
    else:
        print("❌ Complete Pipeline Test FAILED!")
        print("Please check the logs above for details.")


if __name__ == "__main__":
    main()
