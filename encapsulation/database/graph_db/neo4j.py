"""Neo4j Graph Store Implementation
Provides Neo4j graph database operations with BaseIndex interface compatibility
"""


from typing import List, Dict, Any, Optional, Literal, Sequence, Union
from datetime import datetime
import neo4j
import json

from pydantic import Field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging
from encapsulation.database.vector_db.base import BaseIndex, BaseIndexConfig
from core.utils.data_model import Document, GraphData


logger = logging.getLogger(__name__)

neo4j_retry_errors = (
    neo4j.exceptions.ServiceUnavailable,
    neo4j.exceptions.TransientError,
    neo4j.exceptions.WriteServiceUnavailable,
    neo4j.exceptions.ClientError,
)


class Neo4jConfig(BaseIndexConfig):
    """Neo4j Graph Store Configuration Class"""
    type: Literal["neo4j"] = "neo4j"

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

    def build(self) -> "Neo4jGraphStore":
        """Build Neo4j graph store instance"""
        return Neo4jGraphStore(self)



class Neo4jGraphStore(BaseIndex[Neo4jConfig]):
    """Neo4j Graph Store Implementation with BaseIndex interface"""

    def __init__(self, config: Neo4jConfig):
        """Initialize Neo4j graph store"""
        super().__init__(config)
        self._driver = None

        try:
            self._driver: neo4j.Driver = neo4j.GraphDatabase.driver(
                self.config.url,
                auth=(self.config.username, self.config.password)
            )
            logger.info(f"✅ Successfully connected to Neo4j database: {self.config.url}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Neo4j connection: {e}")
            raise

    def _execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query synchronously"""
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")
            logger.error(f"   Query: {query}")
            logger.error(f"   Parameters: {parameters}")
            raise

    def close(self):
        """Close database connection"""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit method"""
        if self._driver:
            self._driver.close()

    # =============================================================================
    # Graph Statistics and Health Check
    # =============================================================================

    def health_check(self) -> Dict[str, Any]:
        """Health check"""
        try:
            # Test database connection
            records = self._execute_query("RETURN 1 as test")
            if not records or records[0]["test"] != 1:
                raise Exception("Database connection test failed")

            # Get basic statistics
            stats = self.get_graph_statistics()

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

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        statistics = {}

        stat_queries = {
            'total_documents': "MATCH (d:Document) RETURN count(d) as count",
            'total_entities': "MATCH (e:Entity) RETURN count(e) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count"
        }

        for stat_name, query in stat_queries.items():
            try:
                result = self._execute_query(query)
                statistics[stat_name] = result[0]['count'] if result else 0
            except Exception as e:
                logger.error(f"⚠️ Error getting statistics {stat_name}: {e}")
                statistics[stat_name] = 0

        return statistics

    # =============================================================================
    # BaseIndex Interface Implementation
    # =============================================================================

    def add(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """Add documents with graph data to Neo4j"""
        return self._add_documents(documents)

    def _add_documents(self, documents: List[Document]) -> List[str]:
        """Synchronous implementation of add documents"""
        added_ids = []

        for doc in documents:
            try:
                # Add document node
                self._add_document(doc)

                # Add graph data if available
                if hasattr(doc, 'graph') and doc.graph:
                    self._add_graph_data(doc.graph, doc.id)

                added_ids.append(doc.id)
                logger.info(f"✅ Successfully added document: {doc.id}")

            except Exception as e:
                logger.error(f"❌ Failed to add document {doc.id}: {e}")
                # Continue with other documents
                continue

        return added_ids

    def _add_document(self, document: Document) -> None:
        """Add document node"""
        query = """
        MERGE (d:Document {id_: $doc_id})
        SET d.content = $content,
            d.metadata = $metadata,
            d.update_time = $update_time,
            d.create_time = CASE WHEN d.create_time IS NULL THEN $create_time ELSE d.create_time END
        RETURN d
        """

        self._execute_query(query, {
            'doc_id': document.id,
            'content': document.content,
            'metadata': json.dumps(document.metadata) if document.metadata else "{}",
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        })

    def _add_graph_data(self, graph_data: GraphData, doc_id: str) -> None:
        """Add graph data"""
        # Add entities
        for entity in graph_data.entities:
            entity_id = doc_id + '_' + entity['id']  # Prefix with doc_id to avoid conflicts
            entity_type = entity.get('entity_type', 'Entity')

            # Prepare entity properties
            properties = {
                'entity_name': entity['entity_name'],
                'entity_type': entity_type,
                'document_id': doc_id,
                'create_time': datetime.now().isoformat(),
                'update_time': datetime.now().isoformat()
            }

            # Add entity attributes
            if 'attributes' in entity and entity['attributes']:
                properties.update(entity['attributes'])

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

            # Create Document-Entity relationship
            doc_entity_query = """
            MATCH (d:Document {id_: $doc_id}), (e:Entity {id_: $entity_id})
            MERGE (d)-[r:MENTIONS]->(e)
            SET r.create_time = $create_time
            RETURN r
            """

            self._execute_query(doc_entity_query, {
                'doc_id': doc_id,
                'entity_id': entity_id,
                'create_time': datetime.now().isoformat()
            })

        # Add relations
        for relation in graph_data.relations:
            if len(relation) >= 3:
                head_name, relation_type, tail_name = relation[0], relation[1], relation[2]

                # Create relationship using entity names
                relation_query = f"""
                MATCH (e1:Entity {{entity_name: $head_name, document_id: $doc_id}}),
                      (e2:Entity {{entity_name: $tail_name, document_id: $doc_id}})
                MERGE (e1)-[r:{relation_type}]->(e2)
                SET r.document_id = $doc_id,
                    r.create_time = $create_time
                RETURN r
                """

                self._execute_query(relation_query, {
                    'head_name': head_name,
                    'tail_name': tail_name,
                    'doc_id': doc_id,
                    'create_time': datetime.now().isoformat()
                })

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents and their graph data"""
        return self._delete_documents(ids)

    def _delete_documents(self, ids: Optional[List[str]] = None) -> bool:
        """Delete documents and their graph data"""
        try:
            if ids is None:
                # Delete all data
                query = "MATCH (n) DETACH DELETE n"
                self._execute_query(query)
                logger.info("✅ Deleted all data from Neo4j")
            else:
                # Delete specific documents and their related data
                for doc_id in ids:
                    # Delete document and all related entities and relationships
                    query = """
                    MATCH (d:Document {id_: $doc_id})
                    OPTIONAL MATCH (d)-[:CONTAINS]->(e:Entity)
                    DETACH DELETE d, e
                    """
                    self._execute_query(query, {'doc_id': doc_id})
                    logger.info(f"✅ Deleted document: {doc_id}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete documents: {e}")
            return False

    def update(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Update documents and their graph data"""
        self._update_documents(documents)





    def _update_documents(self, documents: List[Document]) -> None:
        """Update documents and their graph data"""
        for doc in documents:
            try:
                # Delete existing graph data for this document
                self._delete_documents([doc.id])

                # Add updated document and graph data
                self._add_documents([doc])

                logger.info(f"✅ Successfully updated document: {doc.id}")

            except Exception as e:
                logger.error(f"❌ Failed to update document {doc.id}: {e}")

    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Get documents by IDs"""
        return self._get_documents(list(ids))

    def _get_documents(self, ids: List[str]) -> List[Document]:
        """Synchronous implementation of get documents"""
        documents = []

        for doc_id in ids:
            try:
                # Get document
                doc_query = "MATCH (d:Document {id_: $doc_id}) RETURN d"
                doc_results = self._execute_query(doc_query, {'doc_id': doc_id})

                if not doc_results:
                    logger.warning(f"⚠️ Document not found: {doc_id}")
                    continue

                doc_data = doc_results[0]['d']

                # Create document
                document = Document(
                    content=doc_data.get('content', ''),
                    id=doc_data.get('id_', doc_id),
                    metadata=json.loads(doc_data.get('metadata', '{}'))
                )

                # Get graph data
                document.graph = self._get_graph_data(doc_id)
                documents.append(document)

            except Exception as e:
                logger.error(f"❌ Failed to get document {doc_id}: {e}")
                continue

        return documents

    def _get_graph_data(self, doc_id: str) -> GraphData:
        """Get graph data for document synchronously"""
        # Get entities
        entity_query = """
        MATCH (e:Entity {document_id: $doc_id})
        RETURN e
        """
        entity_results = self._execute_query(entity_query, {'doc_id': doc_id})

        entities = []
        for result in entity_results:
            entity_data = result['e']
            entity = {
                'id': entity_data.get('id_', '').replace(f"{doc_id}_", ""),  # Remove doc_id prefix
                'entity_name': entity_data.get('entity_name', ''),
                'entity_type': entity_data.get('entity_type', ''),
                'attributes': {}
            }

            # Extract attributes (exclude system properties)
            for key, value in entity_data.items():
                if key not in ['id_', 'entity_name', 'entity_type', 'document_id', 'create_time', 'update_time']:
                    entity['attributes'][key] = value

            entities.append(entity)

        # Get relations
        relation_query = """
        MATCH (e1:Entity {document_id: $doc_id})-[r]->(e2:Entity {document_id: $doc_id})
        WHERE r.document_id = $doc_id
        RETURN e1.entity_name as from_name, type(r) as rel_type, e2.entity_name as to_name
        """
        relation_results = self._execute_query(relation_query, {'doc_id': doc_id})

        relations = []
        for result in relation_results:
            relations.append([
                result['from_name'],
                result['rel_type'],
                result['to_name']
            ])

        return GraphData(entities=entities, relations=relations, metadata={})

    def save_index(self, index_path: str, index_name: str = "index") -> None:
        """Save index to disk (Neo4j handles persistence automatically)"""
        logger.info("Neo4j handles data persistence automatically")

    def load_index(self, index_path: Optional[str] = None) -> None:
        """Load index from disk (Neo4j handles persistence automatically)"""
        logger.info("Neo4j handles data persistence automatically")

    def build_index(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Build index from documents"""
        if self.index_exists():
            raise RuntimeError("Index already exists. Use add() to add documents or delete() first.")

        self._add_documents(documents)

    def index_exists(self) -> bool:
        """Check if index exists (check if database is accessible)"""
        try:
            result = self._execute_query("RETURN 1 as test")
            return len(result) > 0
        except Exception:
            return False

    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        statistics = {}

        stat_queries = {
            'total_documents': "MATCH (d:Document) RETURN count(d) as count",
            'total_entities': "MATCH (e:Entity) RETURN count(e) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count"
        }

        for stat_name, query in stat_queries.items():
            try:
                result = self._execute_query(query)
                statistics[stat_name] = result[0]['count'] if result else 0
            except Exception as e:
                logger.error(f"⚠️ Error getting statistics {stat_name}: {e}")
                statistics[stat_name] = 0

        return statistics

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self._get_graph_statistics()
            return {
                'total_documents': stats.get('total_documents', 0),
                'total_entities': stats.get('total_entities', 0),
                'total_relationships': stats.get('total_relationships', 0)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get index stats: {e}")
            return {}

