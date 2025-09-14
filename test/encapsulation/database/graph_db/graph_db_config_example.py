"""Neo4j Graph Database Configuration Injection Usage Example

Demonstrates how to use configuration classes to initialize and use Neo4j graph database.
"""
import os
import sys
from pathlib import Path

# Get project root directory and add to system path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
from encapsulation.database.graph_db.base import BaseNeo4jConfig, GraphStoreBaseNeo4j


async def main():
    """Main function: Demonstrate configuration injection usage"""
    
    # 1. Create configuration object
    config = BaseNeo4jConfig(
        url="bolt://localhost:7681",
        username="neo4j",
        password="12345678",
        database="neo4j"
    )
    
    print("📋 Configuration Information:")
    print(f"  Database URL: {config.url}")
    print(f"  Database Name: {config.database}")
    print(f"  Username: {config.username}")
    print()
    
    graph_store = config.build()
    
    # 5. Health check
    print("\n🏥 Performing health check...")
    health_status = await graph_store.health_check()
    print(f"Health Status: {health_status['status']}")
    if health_status['status'] == 'healthy':
        print(f"Statistics: {health_status['statistics']}")
    else:
        print(f"Error Message: {health_status.get('error', 'Unknown error')}")
    
    # 6. Basic operations example
    if health_status['status'] == 'healthy':
        print("\n📝 Performing basic operations...")
        
        # Create entity
        success = await graph_store.create_entity(
            entity_id="person_001",
            entity_name="Zhang San",
            properties={
                "age": 30,
                "occupation": "Engineer",
                "city": "Beijing"
            }
        )
        print(f"Create entity result: {success}")
        
        # Create another entity
        success = await graph_store.create_entity(
            entity_id="company_001",
            entity_name="Tech Company",
            properties={
                "industry": "Software Development",
                "size": "Medium",
                "location": "Beijing"
            }
        )
        print(f"Create company entity result: {success}")
        
        # Create relationship
        success = await graph_store.create_relationship(
            from_entity_id="person_001",
            to_entity_id="company_001",
            relationship_type="WORKS_FOR",
            properties={
                "start_date": "2020-01-01",
                "position": "Senior Engineer"
            }
        )
        print(f"Create relationship result: {success}")
        
        # Query entity
        entity = await graph_store.get_entity("person_001")
        if entity:
            print(f"Found entity: {entity['entity_name']}")
        
        # Query relationships
        relationships = await graph_store.get_relationships("person_001")
        print(f"Entity relationship count: {len(relationships)}")
        
        # Get statistics
        stats = await graph_store.get_graph_statistics()
        print(f"Graph statistics: {stats}")
    
    # 7. Close connection
    print("\n🔒 Closing database connection...")
    await graph_store.close()
    print("✅ Example completed")


def demo_config_validation():
    """Demonstrate configuration validation functionality"""
    print("\n🔍 Configuration validation example:")
    
    try:
        # Invalid URL
        invalid_config = BaseNeo4jConfig(
            url="invalid://localhost:7687",  # Invalid protocol
            username="neo4j",
            password="password"
        )
    except ValueError as e:
        print(f"❌ URL validation failed: {e}")
    
    
    # Valid configuration
    try:
        valid_config = BaseNeo4jConfig(
            url="bolt://localhost:7687",
            username="neo4j",
            password="password"
        )
        print(f"✅ Configuration validation passed: {valid_config.type}")
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")


if __name__ == "__main__":
    print("🚀 Neo4j Graph Database Configuration Injection Example")
    print("=" * 50)
    
    # Demonstrate configuration validation
    demo_config_validation()
    
    # Run main example
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Example execution failed: {e}")
        print("💡 Please ensure Neo4j database is running and connection configuration is correct")