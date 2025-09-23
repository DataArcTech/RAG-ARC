from typing import List, Dict, Any, Optional, Sequence
from datetime import datetime
import neo4j
import json

from pydantic import Field

import logging
from encapsulation.database.graph_db.base import GraphStore
from encapsulation.data_model.schema import Document, GraphData


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

    def build_index(self, documents: List[Document]) -> None:
        """Build graph from a list of Documents."""

        self.add_documents(documents)

    def update_index(self, documents: List[Document]) -> Optional[bool]:
        """Update existing documents' graphs in the database."""
        try:
            self.update_documents(documents)
            return True
        except Exception as e:
            logger.error(f"Failed to update index: {e}")
            return False

    def add_documents(self, documents: List[Document]) -> List[str]:
        added_ids = []

        # Batch check for existing documents (more efficient for large lists)
        if len(documents) > 1:
            doc_ids = [doc.id for doc in documents]
            batch_check_query = """
            UNWIND $doc_ids as doc_id
            OPTIONAL MATCH (d:Document {id_: doc_id})
            RETURN doc_id, count(d) > 0 as exists
            """
            existing_results = self._execute_query(batch_check_query, {'doc_ids': doc_ids})
            existing_ids = {result['doc_id'] for result in existing_results if result['exists']}
        else:
            existing_ids = set()

        for doc in documents:
            try:
                # Check if document already exists
                if len(documents) > 1:
                    # Use batch check results
                    if doc.id in existing_ids:
                        logger.warning(f"Document with ID {doc.id} already exists, skipping...")
                        continue
                else:
                    # Single document check
                    existing_doc_query = "MATCH (d:Document {id_: $doc_id}) RETURN count(d) as count"
                    existing_results = self._execute_query(existing_doc_query, {'doc_id': doc.id})

                    if existing_results and existing_results[0]['count'] > 0:
                        logger.warning(f"Document with ID {doc.id} already exists, skipping...")
                        continue

                # Add document node
                self.add_document(doc)

                # Add graph data if available
                if hasattr(doc, 'graph') and doc.graph:
                    self.add_graph_data(doc.graph, doc.id)

                added_ids.append(doc.id)
                logger.info(f"Successfully added document: {doc.id}")

            except Exception as e:
                logger.error(f"Failed to add document {doc.id}: {e}")
                # Continue with other documents
                continue

        return added_ids

    def add_document(self, document: Document) -> None:
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
            'metadata': json.dumps(document.metadata, ensure_ascii=False) if document.metadata else "{}",
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        })

    def add_graph_data(self, graph_data: GraphData, doc_id: str) -> None:
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

    def delete_index(self, ids: Optional[List[str]] = None) -> Optional[bool]:
        """Delete documents and their graphs by IDs. Delete all if ids is None."""
        if ids is None:
            raise ValueError("Dangerous operation: delete_index requires specific IDs. Use delete_all_documents() if you want to clear all data.")
        else:
            # Remove duplicates from ids list while preserving order
            unique_ids = list(dict.fromkeys(ids))  # Preserves order, removes duplicates
            if len(unique_ids) != len(ids):
                logger.info(f"Removed {len(ids) - len(unique_ids)} duplicate IDs from delete list")
            return self.delete_documents(unique_ids)

    def delete_documents(self, ids: Optional[List[str]] = None) -> bool:
        """Delete documents and their graph data"""
        try:
            if not ids:
                raise ValueError("Must provide document IDs to delete")

            # Remove duplicates from ids list while preserving order
            unique_ids = list(dict.fromkeys(ids))
            if len(unique_ids) != len(ids):
                logger.info(f"Removed {len(ids) - len(unique_ids)} duplicate IDs from delete list")

            # Delete specific documents and their related data
            for doc_id in unique_ids:
                # Delete document and all related entities and relationships
                query = """
                MATCH (d:Document {id_: $doc_id})
                OPTIONAL MATCH (d)-[r:MENTIONS]->(e:Entity)
                DETACH DELETE d, r
                """
                self._execute_query(query, {'doc_id': doc_id})
                logger.info(f"Deleted document: {doc_id}")

            clean_orphans_query = """
            MATCH (e:Entity)
            WHERE NOT (e)-[:MENTIONS]-(:Document)
            DELETE e
            """

            self._execute_query(clean_orphans_query)
            logger.info("Cleaned up orphan entities with no MENTIONS")

            return True

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all documents and their graph data"""
        if not confirm:
            raise ValueError(" Dangerous operation: delete_all_documents requires confirm=True")
        try:
            query = "MATCH (n) DETACH DELETE n"
            self._execute_query(query)
            logger.info("Deleted all data from Neo4j")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all data: {e}")
            return False


    def update_documents(self, documents: List[Document]) -> None:
        """Update documents and their graph data"""
        for doc in documents:
            try:
                # Delete existing graph data for this document
                self.delete_documents([doc.id])

                # Add updated document and graph data
                self.add_documents([doc])

                logger.info(f"Successfully updated document: {doc.id}")

            except Exception as e:
                logger.error(f"Failed to update document {doc.id}: {e}")

    def get_by_ids(self, ids: Sequence[str]) -> List[Document]:
        """Get documents by IDs"""
        return self.get_documents(list(ids))

    def get_documents(self, ids: List[str]) -> List[Document]:
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
                document.graph = self.get_graph_data(doc_id)
                documents.append(document)

            except Exception as e:
                logger.error(f"Failed to get document {doc_id}: {e}")
                continue

        return documents

    def get_graph_data(self, doc_id: str) -> GraphData:
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
                'attributes': json.loads(entity_data.get('attributes', '{}'))
            }

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
            'total_documents': "MATCH (d:Document) RETURN count(d) as count",
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