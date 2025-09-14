"""Neo4j Graph Store Base Class
Provides basic operations for Neo4j graph database
"""

import asyncio
from typing import List, Dict, Any, Optional, TypeVar, Generic, Literal
from datetime import datetime
import neo4j

from pydantic import Field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging
from framework.module import AbstractModule
from framework.config import AbstractConfig


logger = logging.getLogger(__name__)

neo4j_retry_errors = (
    neo4j.exceptions.ServiceUnavailable,
    neo4j.exceptions.TransientError,
    neo4j.exceptions.WriteServiceUnavailable,
    neo4j.exceptions.ClientError,
)


ConfigType = TypeVar("ConfigType", bound="BaseNeo4jConfig")

class BaseNeo4jConfig(AbstractConfig):
    """Neo4j Graph Store Configuration Class
    
    Defines basic configuration parameters for Neo4j graph database connection
    
    Attributes:
        type: Configuration type identifier
        url: Neo4j database connection URL
        username: Database username
        password: Database password
        database: Database name
    """
    type: Literal["base_neo4j"] = "base_neo4j"
    
    # Database connection configuration
    url: str = Field(
        description="Neo4j database connection URL, e.g.: bolt://localhost:7687"
    )
    username: str = Field(
        description="Database username"
    )
    password: str = Field(
        description="Database password"
    )
    database: str = Field(
        default="neo4j",
        description="Database name"
    )
    
    
    
    def build(self) -> "GraphStoreBaseNeo4j":
        """Build Neo4j graph store instance
        
        Returns:
            GraphStoreBaseNeo4j: Configured Neo4j graph store instance
        """
        from .base import GraphStoreBaseNeo4j
        return GraphStoreBaseNeo4j(config=self)



class GraphStoreBaseNeo4j(AbstractModule,Generic[ConfigType]):
    """Neo4j Graph Store Base Class
    Provides basic connection and query execution functionality for Neo4j graph database
    
    Supports configuration injection, initialized through GraphStoreNeo4jConfig configuration object.
    Provides basic CRUD operations for entities and relationships, as well as general Cypher query execution functionality.
    
    Features:
    - Configuration-driven connection management
    - Configurable retry mechanism
    - Automatic constraint and index creation
    - Complete entity and relationship operation API
    - Health check and statistics
    """
    config: ConfigType
    
    def __init__(self, config: ConfigType):
        """Initialize Neo4j graph store base class
        
        Args:
            config: Neo4j graph store configuration object
        """
        super().__init__(config)
        self._driver = None
        self._driver_lock = asyncio.Lock()
        
        try:
            self._driver: neo4j.AsyncDriver = neo4j.AsyncGraphDatabase.driver(
                self.config.url, 
                auth=(self.config.username, self.config.password)
            )
            logger.info(f"✅ Successfully connected to Neo4j database: {self.config.url}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j connection: {e}")
            raise

    async def close(self):
        """Close database connection"""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit method"""
        if self._driver:
            await self._driver.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(neo4j_retry_errors))
    async def _execute_query(self, query: str, parameters: Dict[str, Any] = None):
        """General method for executing Neo4j queries with retry mechanism
        
        Args:
            query: Cypher query statement
            parameters: Query parameters
            
        Returns:
            Query result data list
        """
        if parameters is None:
            parameters = {}
            
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, **parameters)
            return await result.data()

    # =============================================================================
    # Entity Operations
    # =============================================================================
    
    async def create_entity(self, entity_id: str, entity_name: str, properties: Dict[str, Any] = None) -> bool:
        """Create entity
        
        Args:
            entity_id: Entity ID
            entity_name: Entity name
            properties: Entity properties
            
        Returns:
            bool: Whether creation was successful
        """
        if properties is None:
            properties = {}
            
        properties.update({
            'id_': entity_id,
            'entity_name': entity_name,
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        })
        
        query = """
        MERGE (e:Entity {id_: $entity_id})
        SET e += $properties
        RETURN e
        """
        
        try:
            await self._execute_query(query, {
                'entity_id': entity_id,
                'properties': properties
            })
            logger.info(f"✅ Successfully created entity: {entity_name} ({entity_id})")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create entity: {e}")
            return False

    async def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """Update entity properties
        
        Args:
            entity_id: Entity ID
            properties: Properties to update
            
        Returns:
            bool: Whether update was successful
        """
        properties['update_time'] = datetime.now().isoformat()
        
        query = """
        MATCH (e:Entity {id_: $entity_id})
        SET e += $properties
        RETURN e
        """
        
        try:
            records = await self._execute_query(query, {
                'entity_id': entity_id,
                'properties': properties
            })
            if records:
                logger.info(f"✅ Successfully updated entity: {entity_id}")
                return True
            else:
                logger.warning(f"⚠️ Entity does not exist: {entity_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to update entity: {e}")
            return False

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete entity
        
        Args:
            entity_id: Entity ID
            
        Returns:
            bool: Whether deletion was successful
        """
        query = """
        MATCH (e:Entity {id_: $entity_id})
        DETACH DELETE e
        RETURN count(e) as deleted_count
        """
        
        try:
            records = await self._execute_query(query, {'entity_id': entity_id})
            deleted_count = records[0]['deleted_count'] if records else 0
            
            if deleted_count > 0:
                logger.info(f"✅ Successfully deleted entity: {entity_id}")
                return True
            else:
                logger.warning(f"⚠️ Entity does not exist: {entity_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to delete entity: {e}")
            return False

    async def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity information
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity information dictionary, or None if not found
        """
        query = """
        MATCH (e:Entity {id_: $entity_id})
        RETURN e
        """
        
        try:
            records = await self._execute_query(query, {'entity_id': entity_id})
            if records:
                return dict(records[0]['e'])
            return None
        except Exception as e:
            logger.error(f"❌ Failed to get entity: {e}")
            return None

    # =============================================================================
    # Relationship Operations
    # =============================================================================
    
    async def create_relationship(self, from_entity_id: str, to_entity_id: str, 
                                relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        """Create relationship
        
        Args:
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            relationship_type: Relationship type
            properties: Relationship properties
            
        Returns:
            bool: Whether creation was successful
        """
        if properties is None:
            properties = {}
            
        properties.update({
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        })
        
        query = f"""
        MATCH (from:Entity {{id_: $from_id}}), (to:Entity {{id_: $to_id}})
        MERGE (from)-[r:{relationship_type}]->(to)
        SET r += $properties
        RETURN r
        """
        
        try:
            await self._execute_query(query, {
                'from_id': from_entity_id,
                'to_id': to_entity_id,
                'properties': properties
            })
            logger.info(f"✅ Successfully created relationship: {from_entity_id} -[{relationship_type}]-> {to_entity_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create relationship: {e}")
            return False

    async def update_relationship(self, from_entity_id: str, to_entity_id: str, 
                                relationship_type: str, properties: Dict[str, Any]) -> bool:
        """Update relationship properties
        
        Args:
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            relationship_type: Relationship type
            properties: Properties to update
            
        Returns:
            bool: Whether update was successful
        """
        properties['update_time'] = datetime.now().isoformat()
        
        query = f"""
        MATCH (from:Entity {{id_: $from_id}})-[r:{relationship_type}]->(to:Entity {{id_: $to_id}})
        SET r += $properties
        RETURN r
        """
        
        try:
            records = await self._execute_query(query, {
                'from_id': from_entity_id,
                'to_id': to_entity_id,
                'properties': properties
            })
            if records:
                logger.info(f"✅ Successfully updated relationship: {from_entity_id} -[{relationship_type}]-> {to_entity_id}")
                return True
            else:
                logger.warning(f"⚠️ Relationship does not exist: {from_entity_id} -[{relationship_type}]-> {to_entity_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to update relationship: {e}")
            return False

    async def delete_relationship(self, from_entity_id: str, to_entity_id: str, 
                                relationship_type: str) -> bool:
        """Delete relationship
        
        Args:
            from_entity_id: Source entity ID
            to_entity_id: Target entity ID
            relationship_type: Relationship type
            
        Returns:
            bool: Whether deletion was successful
        """
        query = f"""
        MATCH (from:Entity {{id_: $from_id}})-[r:{relationship_type}]->(to:Entity {{id_: $to_id}})
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        try:
            records = await self._execute_query(query, {
                'from_id': from_entity_id,
                'to_id': to_entity_id
            })
            deleted_count = records[0]['deleted_count'] if records else 0
            
            if deleted_count > 0:
                logger.info(f"✅ Successfully deleted relationship: {from_entity_id} -[{relationship_type}]-> {to_entity_id}")
                return True
            else:
                logger.warning(f"⚠️ Relationship does not exist: {from_entity_id} -[{relationship_type}]-> {to_entity_id}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to delete relationship: {e}")
            return False

    async def get_relationships(self, entity_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """Get entity relationships
        
        Args:
            entity_id: Entity ID
            direction: Relationship direction ("in", "out", "both")
            
        Returns:
            List of relationships
        """
        if direction == "out":
            query = """
            MATCH (e:Entity {id_: $entity_id})-[r]->(target)
            RETURN type(r) as relationship_type, properties(r) as properties, 
                   target.id_ as target_id, target.entity_name as target_name
            """
        elif direction == "in":
            query = """
            MATCH (source)-[r]->(e:Entity {id_: $entity_id})
            RETURN type(r) as relationship_type, properties(r) as properties,
                   source.id_ as source_id, source.entity_name as source_name
            """
        else:  # both
            query = """
            MATCH (e:Entity {id_: $entity_id})-[r]->(target)
            RETURN 'out' as direction, type(r) as relationship_type, properties(r) as properties,
                   target.id_ as other_id, target.entity_name as other_name
            UNION
            MATCH (source)-[r]->(e:Entity {id_: $entity_id})
            RETURN 'in' as direction, type(r) as relationship_type, properties(r) as properties,
                   source.id_ as other_id, source.entity_name as other_name
            """
        
        try:
            records = await self._execute_query(query, {'entity_id': entity_id})
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"❌ Failed to get relationships: {e}")
            return []

    # =============================================================================
    # General Query
    # =============================================================================
    
    async def execute_cypher(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute custom Cypher query
        
        Args:
            query: Cypher query statement
            parameters: Query parameters
            
        Returns:
            Query result list
        """
        try:
            records = await self._execute_query(query, parameters)
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"❌ Failed to execute query: {e}")
            return []

    async def get_graph_statistics(self) -> Dict[str, int]:
        """Get graph statistics"""
        queries = {
            'total_entities': 'MATCH (e:Entity) RETURN count(e) as count',
            'total_relationships': 'MATCH ()-[r]->() RETURN count(r) as count'
        }
        
        statistics = {}
        for stat_name, query in queries.items():
            try:
                records = await self._execute_query(query)
                if records:
                    statistics[stat_name] = records[0]["count"]
                else:
                    statistics[stat_name] = 0
            except Exception as e:
                logger.error(f"⚠️ Error getting statistics {stat_name}: {e}")
                statistics[stat_name] = 0
        
        return statistics

    async def health_check(self) -> Dict[str, Any]:
        """Health check"""
        try:
            # Test database connection
            records = await self._execute_query("RETURN 1 as test")
            if not records or records[0]["test"] != 1:
                raise Exception("Database connection test failed")
            
            # Get basic statistics
            stats = await self.get_graph_statistics()
            
            return {
                "status": "healthy",
                "database": self.config.database,
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # =============================================================================
    # Abstract Methods - Optional Implementation by Subclasses
    # =============================================================================
    
    async def create_constraints_and_indexes(self):
        """Create database constraints and indexes (optional implementation by subclasses)"""
        # Create basic constraints
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id_ IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                await self._execute_query(constraint)
                logger.info(f"✅ Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create constraint: {e}")
        
        # Create basic indexes
        indexes = [
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_name)"
        ]
        
        for index in indexes:
            try:
                await self._execute_query(index)
                logger.info(f"✅ Created index: {index}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create index: {e}")
    
    @classmethod
    def from_config(cls, config: ConfigType) -> "GraphStoreBaseNeo4j":
        """Create instance from configuration object
        
        Args:
            config: Neo4j graph store configuration object
            
        Returns:
            GraphStoreBaseNeo4j: Configured instance
        """
        return cls(config=config)
    
    @classmethod
    def class_name(cls) -> str:
        """Return class name"""
        return cls.__name__