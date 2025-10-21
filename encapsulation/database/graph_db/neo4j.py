from typing import List, Dict, Any, Optional, Sequence
from datetime import datetime
import neo4j
import json

from pydantic import Field

import logging
from encapsulation.database.graph_db.base import GraphStore
from encapsulation.data_model.schema import Chunk, GraphData


logger = logging.getLogger(__name__)

neo4j_retry_errors = (
    neo4j.exceptions.ServiceUnavailable,
    neo4j.exceptions.TransientError,
    neo4j.exceptions.WriteServiceUnavailable,
    neo4j.exceptions.ClientError,
)


class Neo4jGraphStore(GraphStore):
    """Neo4j Graph Store Implementation with BaseIndex interface"""

    def __init__(self, config):
        """Initialize Neo4j graph store"""
        self.config = config
        self._driver = None

        try:
            self._driver: neo4j.Driver = neo4j.GraphDatabase.driver(
                self.config.url,
                auth=(self.config.username, self.config.password)
            )
            logger.info(f"Successfully connected to Neo4j database: {self.config.url}")

        except Exception as e:
            logger.error(f"Failed to initialize Neo4j connection: {e}")
            raise

    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query synchronously"""
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"   Query: {query}")
            logger.error(f"   Parameters: {parameters}")
            raise

    def close(self):
        """Close database connection"""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit method"""
        if self._driver:
            self._driver.close()

    # =============================================================================
    # BaseIndex Interface Implementation
    # =============================================================================

    def build_index(self, chunks: List[Chunk]) -> List[str]:
        """Build graph from a list of Chunks."""

        return self.add_chunks(chunks)

    def update_index(self, chunks: List[Chunk]) -> Optional[bool]:
        """Update existing chunks' graphs in the database."""
        try:
            self.update_chunks(chunks)
            return True
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return False

    def add_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Add multiple chunks to the graph"""
        added_ids = []

        # Batch check for existing chunks (more efficient for large lists)
        if len(chunks) > 1:
            chunk_ids = [chunk.id for chunk in chunks]
            batch_check_query = """
            UNWIND $chunk_ids as chunk_id
            OPTIONAL MATCH (c:Chunk {id_: chunk_id})
            RETURN chunk_id, count(c) > 0 as exists
            """
            existing_results = self._execute_query(batch_check_query, {'chunk_ids': chunk_ids})
            existing_ids = {result['chunk_id'] for result in existing_results if result['exists']}
        else:
            existing_ids = set()

        for chunk in chunks:
            try:
                # Check if chunk already exists
                if len(chunks) > 1:
                    # Use batch check results
                    if chunk.id in existing_ids:
                        logger.warning(f"Chunk with ID {chunk.id} already exists, skipping...")
                        continue
                else:
                    # Single chunk check
                    existing_chunk_query = "MATCH (c:Chunk {id_: $chunk_id}) RETURN count(c) as count"
                    existing_results = self._execute_query(existing_chunk_query, {'chunk_id': chunk.id})

                    if existing_results and existing_results[0]['count'] > 0:
                        logger.warning(f"Chunk with ID {chunk.id} already exists, skipping...")
                        continue

                # Add chunk node
                self.add_chunk(chunk)

                # Add graph data if available
                if hasattr(chunk, 'graph') and chunk.graph:
                    self.add_graph_data(chunk.graph, chunk.id)

                added_ids.append(chunk.id)
                logger.info(f"Successfully added chunk: {chunk.id}")

            except Exception as e:
                logger.error(f"Failed to add chunk {chunk.id}: {e}")
                # Continue with other chunks
                continue

        return added_ids

    def add_chunk(self, chunk: Chunk) -> None:
        """Add chunk node"""
        query = """
        MERGE (c:Chunk {id_: $chunk_id})
        SET c.content = $content,
            c.metadata = $metadata,
            c.update_time = $update_time,
            c.create_time = CASE WHEN c.create_time IS NULL THEN $create_time ELSE c.create_time END
        RETURN c
        """

        self._execute_query(query, {
            'chunk_id': chunk.id,
            'content': chunk.content,
            'metadata': json.dumps(chunk.metadata, ensure_ascii=False) if chunk.metadata else "{}",
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        })

    def add_graph_data(self, graph_data: GraphData, chunk_id: str) -> None:
        """Add graph data"""
        # Add entities
        for entity in graph_data.entities:
            entity_id = chunk_id + '_' + entity['id']  # Prefix with chunk_id to avoid conflicts
            entity_type = entity.get('entity_type', 'Entity')

            # Prepare entity properties
            properties = {
                'entity_name': entity['entity_name'],
                'entity_type': entity_type,
                'chunk_id': chunk_id,
                'create_time': datetime.now().isoformat(),
                'update_time': datetime.now().isoformat(),
                'attributes': json.dumps(entity['attributes'], ensure_ascii=False) if entity.get('attributes') else "{}"
            }

            # Create entity with dynamic label
            query = f"""
            MERGE (e:Entity:{entity_type} {{id_: $entity_id}})
            SET e += $properties
            RETURN e
            """

            self._execute_query(query, {
                'entity_id': entity_id,
                'properties': properties
            })

            # Create Chunk-Entity relationship
            chunk_entity_query = """
            MATCH (c:Chunk {id_: $chunk_id}), (e:Entity {id_: $entity_id})
            MERGE (c)-[r:MENTIONS]->(e)
            SET r.create_time = $create_time
            RETURN r
            """

            self._execute_query(chunk_entity_query, {
                'chunk_id': chunk_id,
                'entity_id': entity_id,
                'create_time': datetime.now().isoformat()
            })

        # Add relations
        for relation in graph_data.relations:
            if len(relation) >= 3:
                head_name, relation_type, tail_name = relation[0], relation[1], relation[2]

                # Create relationship using entity names
                relation_query = f"""
                MATCH (e1:Entity {{entity_name: $head_name, chunk_id: $chunk_id}}),
                      (e2:Entity {{entity_name: $tail_name, chunk_id: $chunk_id}})
                MERGE (e1)-[r:{relation_type}]->(e2)
                SET r.chunk_id = $chunk_id,
                    r.create_time = $create_time
                RETURN r
                """

                self._execute_query(relation_query, {
                    'head_name': head_name,
                    'tail_name': tail_name,
                    'chunk_id': chunk_id,
                    'create_time': datetime.now().isoformat()
                })

    def delete_index(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        """Delete chunks and their graphs by IDs. Delete all if ids is None."""
        if ids is None:
            raise ValueError("Dangerous operation: delete_index requires specific IDs. Use delete_all_chunks() if you want to clear all data.")
        else:
            # Remove duplicates from ids list while preserving order
            unique_ids = list(dict.fromkeys(ids))  # Preserves order, removes duplicates
            if len(unique_ids) != len(ids):
                logger.info(f"Removed {len(ids) - len(unique_ids)} duplicate IDs from delete list")
            return self.delete_chunks(unique_ids)

    def delete_chunks(self, ids: Optional[List[str]] = None) -> bool:
        """Delete chunks and their graph data"""
        try:
            if not ids:
                raise ValueError("Must provide chunk IDs to delete")

            # Remove duplicates from ids list while preserving order
            unique_ids = list(dict.fromkeys(ids))
            if len(unique_ids) != len(ids):
                logger.info(f"Removed {len(ids) - len(unique_ids)} duplicate IDs from delete list")

            # Delete specific chunks and their related data
            for chunk_id in unique_ids:
                # Delete chunk and all related entities and relationships
                query = """
                MATCH (c:Chunk {id_: $chunk_id})
                OPTIONAL MATCH (c)-[r:MENTIONS]->(e:Entity)
                DETACH DELETE c, r
                """
                self._execute_query(query, {'chunk_id': chunk_id})
                logger.info(f"Deleted chunk: {chunk_id}")

            clean_orphans_query = """
            MATCH (e:Entity)
            WHERE NOT (e)-[:MENTIONS]-(:Chunk)
            DELETE e
            """

            self._execute_query(clean_orphans_query)
            logger.info("Cleaned up orphan entities with no MENTIONS")

            return True

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all chunks and their graph data"""
        if not confirm:
            raise ValueError(" Dangerous operation: delete_all_chunks requires confirm=True")
        try:
            query = "MATCH (n) DETACH DELETE n"
            self._execute_query(query)
            logger.info("Deleted all data from Neo4j")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all data: {e}")
            return False


    def update_chunks(self, chunks: List[Chunk]) -> None:
        """Update chunks and their graph data"""
        for chunk in chunks:
            try:
                # Delete existing graph data for this chunk
                self.delete_index([chunk.id])

                # Add updated chunk and graph data
                self.add_chunks([chunk])

                logger.info(f"Successfully updated chunk: {chunk.id}")

            except Exception as e:
                logger.error(f"Failed to update chunk {chunk.id}: {e}")

    def get_by_ids(self, ids: Sequence[str]) -> List[Chunk]:
        """Retrieve chunks (including their graphs) by IDs."""
        ids_list = list(ids)
        return self._get_chunks_impl(ids_list)

    def get_chunks(self, ids: List[str]) -> List[Chunk]:
        """Deprecated: Use get_by_ids instead"""
        return self._get_chunks_impl(ids)

    def _get_chunks_impl(self, ids: List[str]) -> List[Chunk]:
        chunks = []

        for chunk_id in ids:
            try:
                # Get chunk
                chunk_query = "MATCH (c:Chunk {id_: $chunk_id}) RETURN c"
                chunk_results = self._execute_query(chunk_query, {'chunk_id': chunk_id})

                if not chunk_results:
                    logger.warning(f"⚠️ Chunk not found: {chunk_id}")
                    continue

                chunk_data = chunk_results[0]['c']

                # Create chunk
                chunk = Chunk(
                    content=chunk_data.get('content', ''),
                    id=chunk_data.get('id_', chunk_id),
                    metadata=json.loads(chunk_data.get('metadata') or '{}')
                )

                # Get graph data
                chunk.graph = self.get_graph_data(chunk_id)
                chunks.append(chunk)

            except Exception as e:
                logger.error(f"Failed to get chunk {chunk_id}: {e}")
                continue

        return chunks

    def get_graph_data(self, chunk_id: str) -> GraphData:
        """Get graph data for chunk synchronously"""
        # Get entities
        entity_query = """
        MATCH (e:Entity {chunk_id: $chunk_id})
        RETURN e
        """
        entity_results = self._execute_query(entity_query, {'chunk_id': chunk_id})

        entities = []
        for result in entity_results:
            entity_data = result['e']
            entity = {
                'id': entity_data.get('id_', '').replace(f"{chunk_id}_", ""),  # Remove chunk_id prefix
                'entity_name': entity_data.get('entity_name', ''),
                'entity_type': entity_data.get('entity_type', ''),
                'attributes': json.loads(entity_data.get('attributes') or '{}')
            }

            entities.append(entity)

        # Get relations
        relation_query = """
        MATCH (e1:Entity {chunk_id: $chunk_id})-[r]->(e2:Entity {chunk_id: $chunk_id})
        WHERE r.chunk_id = $chunk_id
        RETURN e1.entity_name as from_name, type(r) as rel_type, e2.entity_name as to_name
        """
        relation_results = self._execute_query(relation_query, {'chunk_id': chunk_id})

        relations = []
        for result in relation_results:
            relations.append([
                result['from_name'],
                result['rel_type'],
                result['to_name']
            ])

        return GraphData(entities=entities, relations=relations, metadata={})

    def save_index(self, path: str, name: str = "index") -> None:
        """Persist the graph database to filesystem."""
        logger.info(f"Neo4j handles data persistence automatically. Path: {path}, Name: {name}")

    def load_index(self, path: str) -> None:
        """Load persisted graph database from filesystem."""
        logger.info(f"Neo4j handles data persistence automatically. Path: {path}")

    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Run a query on the graph database.Use this method to run any Cypher query on the graph database."""
        return self._execute_query(query, params)



    def health_check(self) -> Dict[str, Any]:
        """Health check"""
        try:
            # Test database connection
            records = self._execute_query("RETURN 1 as test")
            if not records or records[0]["test"] != 1:
                raise Exception("Database connection test failed")

            # Get basic statistics
            stats = self.get_graph_db_info()

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

    def get_graph_db_info(self) -> Dict[str, Any]:
        """Return statistics or metadata about the graph database."""
        statistics = {}

        stat_queries = {
            'total_chunks': "MATCH (c:Chunk) RETURN count(c) as count",
            'total_entities': "MATCH (e:Entity) RETURN count(e) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count"
        }

        for stat_name, query in stat_queries.items():
            try:
                result = self._execute_query(query)
                statistics[stat_name] = result[0]['count'] if result else 0
            except Exception as e:
                logger.error(f"Error getting statistics {stat_name}: {e}")
                statistics[stat_name] = 0

        return statistics