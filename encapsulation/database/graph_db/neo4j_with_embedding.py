from typing import Any, List, Dict, Annotated, Union, Literal
from datetime import datetime
import json
from pydantic import Field
from encapsulation.database.graph_db.neo4j import Neo4jGraphStore
from encapsulation.data_model.schema import Chunk, GraphData
import logging

logger = logging.getLogger(__name__)




class Neo4jVectorGraphStore(Neo4jGraphStore):
    """Neo4j Vector Graph Store with embedding support for Chunks and Entities"""

    def __init__(self, config):
        """Initialize Neo4j vector graph store with embedding model"""
        super().__init__(config)
        self.embedding_model = config.embedding.build()
        logger.info("Neo4j Vector Graph Store initialized with embedding support")

    def generate_chunk_embedding(self, chunk: Chunk) -> List[float]:
        """Generate embedding for chunk content"""
        try:
            if not chunk.content:
                return []
            embedding = self.embedding_model.embed(chunk.content)
            return embedding if isinstance(embedding, list) else embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate chunk embedding: {e}")
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

            embedding = self.embedding_model.embed(text_for_embedding)
            return embedding if isinstance(embedding, list) else embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate entity embedding: {e}")
            return []

    def add_chunk(self, chunk: Chunk) -> None:
        """Add chunk node with embedding"""
        # Generate embedding for chunk content
        embedding = self.generate_chunk_embedding(chunk)

        query = """
        MERGE (c:Chunk {id_: $chunk_id})
        SET c.content = $content,
            c.metadata = $metadata,
            c.embedding = $embedding,
            c.update_time = $update_time,
            c.create_time = CASE WHEN c.create_time IS NULL THEN $create_time ELSE c.create_time END
        RETURN c
        """

        self._execute_query(query, {
            'chunk_id': chunk.id,
            'content': chunk.content,
            'metadata': json.dumps(chunk.metadata, ensure_ascii=False) if chunk.metadata else "{}",
            'embedding': embedding,
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        })

    def add_graph_data(self, graph_data: GraphData, chunk_id: str) -> None:
        """Add graph data with entity embeddings"""
        # Add entities with embeddings
        for entity in graph_data.entities:
            entity_id = chunk_id + '_' + entity['id']  # Prefix with chunk_id to avoid conflicts
            entity_type = entity.get('entity_type', 'Entity')

            # Generate embedding for entity
            entity_embedding = self.generate_entity_embedding(entity)

            # Prepare entity properties
            properties = {
                'entity_name': entity['entity_name'],
                'entity_type': entity_type,
                'chunk_id': chunk_id,
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

        # Add relations (same as parent class)
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

    def get_chunks(self, ids: List[str]) -> List[Chunk]:
        """Get chunks with embeddings"""
        chunks = []

        for chunk_id in ids:
            try:
                # Get Chunk with embedding
                chunk_query = "MATCH (c:Chunk {id_: $chunk_id}) RETURN c"
                chunk_results = self._execute_query(chunk_query, {'chunk_id': chunk_id})

                if not chunk_results:
                    logger.warning(f"Chunk not found: {chunk_id}")
                    continue

                chunk_data = chunk_results[0]['c']

                # Create chunk with embedding in metadata
                metadata = json.loads(chunk_data.get('metadata') or '{}')
                if chunk_data.get('embedding'):
                    metadata['embedding'] = chunk_data['embedding']

                chunk = Chunk(
                    content=chunk_data.get('content', ''),
                    id=chunk_data.get('id_', chunk_id),
                    metadata=metadata
                )

                # Get graph data with entity embeddings
                chunk.graph = self.get_graph_data(chunk_id)
                chunks.append(chunk)

            except Exception as e:
                logger.error(f"Failed to get chunk {chunk_id}: {e}")
                continue

        return chunks

    def get_graph_data(self, chunk_id: str) -> GraphData:
        """Get graph data with entity embeddings"""
        # Get entities with embeddings
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

            # Add embedding to attributes if available
            if entity_data.get('embedding'):
                entity['attributes']['embedding'] = entity_data['embedding']

            entities.append(entity)

        # Get relations (same as parent class)
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

    def delete_all_index(self, confirm: bool = False) -> bool:
        """Delete all chunks and their graph data - implements abstract method"""
        return self.delete_all_index(confirm)