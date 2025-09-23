from typing import Any, List, Dict, Annotated, Union, Literal
from datetime import datetime
import json
from pydantic import Field
from encapsulation.database.graph_db.neo4j import Neo4jGraphStore
from encapsulation.data_model.schema import Document, GraphData
import logging

logger = logging.getLogger(__name__)




class Neo4jVectorGraphStore(Neo4jGraphStore):
    """Neo4j Vector Graph Store with embedding support for Documents and Entities"""

    def __init__(self, config):
        """Initialize Neo4j vector graph store with embedding model"""
        super().__init__(config)
        self.embedding_model = config.embedding.build()
        logger.info("Neo4j Vector Graph Store initialized with embedding support")

    def generate_document_embedding(self, document: Document) -> List[float]:
        """Generate embedding for document content"""
        try:
            if not document.content:
                return []
            embedding = self.embedding_model.embed_query(document.content)
            return embedding if isinstance(embedding, list) else embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate document embedding: {e}")
            return []

    def generate_entity_embedding(self, entity: Dict[str, Any]) -> List[float]:
        """Generate embedding for entity from entity_name + attributes"""
        try:
            entity_name = entity.get('entity_name', '')
            attributes = entity.get('attributes', {})

            # Serialize attributes to string for embedding (preserve Chinese characters)
            attributes_str = json.dumps(attributes, sort_keys=True, ensure_ascii=False) if attributes else ""

            # Combine entity name and attributes
            text_for_embedding = f"{entity_name} {attributes_str}".strip()

            if not text_for_embedding:
                return []

            embedding = self.embedding_model.embed_query(text_for_embedding)
            return embedding if isinstance(embedding, list) else embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate entity embedding: {e}")
            return []

    def add_document(self, document: Document) -> None:
        """Add document node with embedding"""
        # Generate embedding for document content
        embedding = self.generate_document_embedding(document)

        query = """
        MERGE (d:Document {id_: $doc_id})
        SET d.content = $content,
            d.metadata = $metadata,
            d.embedding = $embedding,
            d.update_time = $update_time,
            d.create_time = CASE WHEN d.create_time IS NULL THEN $create_time ELSE d.create_time END
        RETURN d
        """

        self._execute_query(query, {
            'doc_id': document.id,
            'content': document.content,
            'metadata': json.dumps(document.metadata, ensure_ascii=False) if document.metadata else "{}",
            'embedding': embedding,
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        })

    def add_graph_data(self, graph_data: GraphData, doc_id: str) -> None:
        """Add graph data with entity embeddings"""
        # Add entities with embeddings
        for entity in graph_data.entities:
            entity_id = doc_id + '_' + entity['id']  # Prefix with doc_id to avoid conflicts
            entity_type = entity.get('entity_type', 'Entity')

            # Generate embedding for entity
            entity_embedding = self.generate_entity_embedding(entity)

            # Prepare entity properties
            properties = {
                'entity_name': entity['entity_name'],
                'entity_type': entity_type,
                'document_id': doc_id,
                'create_time': datetime.now().isoformat(),
                'update_time': datetime.now().isoformat(),
                'attributes': json.dumps(entity['attributes'], ensure_ascii=False) if entity.get('attributes') else "{}",
                'embedding': entity_embedding
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

        # Add relations (same as parent class)
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

    def get_documents(self, ids: List[str]) -> List[Document]:
        """Get documents with embeddings"""
        documents = []

        for doc_id in ids:
            try:
                # Get document with embedding
                doc_query = "MATCH (d:Document {id_: $doc_id}) RETURN d"
                doc_results = self._execute_query(doc_query, {'doc_id': doc_id})

                if not doc_results:
                    logger.warning(f"Document not found: {doc_id}")
                    continue

                doc_data = doc_results[0]['d']

                # Create document with embedding in metadata
                metadata = json.loads(doc_data.get('metadata', '{}'))
                if doc_data.get('embedding'):
                    metadata['embedding'] = doc_data['embedding']

                document = Document(
                    content=doc_data.get('content', ''),
                    id=doc_data.get('id_', doc_id),
                    metadata=metadata
                )

                # Get graph data with entity embeddings
                document.graph = self.get_graph_data(doc_id)
                documents.append(document)

            except Exception as e:
                logger.error(f"Failed to get document {doc_id}: {e}")
                continue

        return documents

    def get_graph_data(self, doc_id: str) -> GraphData:
        """Get graph data with entity embeddings"""
        # Get entities with embeddings
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

            # Add embedding to attributes if available
            if entity_data.get('embedding'):
                entity['attributes']['embedding'] = entity_data['embedding']

            entities.append(entity)

        # Get relations (same as parent class)
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

    def delete_all_Index(self, confirm: bool = False) -> bool:
        """Delete all documents and their graph data - implements abstract method"""
        return self.delete_all_index(confirm)