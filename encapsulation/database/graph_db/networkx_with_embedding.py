from typing import Any, List, Dict
from datetime import datetime
import json
import logging

from encapsulation.database.graph_db.networkx_graph import NetworkXGraphStore
from encapsulation.data_model.schema import Document, GraphData

logger = logging.getLogger(__name__)


class NetworkXVectorGraphStore(NetworkXGraphStore):
    """NetworkX Vector Graph Store with embedding support for Documents and Entities"""

    def __init__(self, config):
        """Initialize NetworkX vector graph store with embedding model"""
        super().__init__(config)
        self.embedding_model = config.embedding.build()

        # Vector search configuration
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.7)
        self.max_search_results = getattr(config, 'max_search_results', 100)

        # Embedding caching configuration
        self.cache_embeddings = getattr(config, 'cache_embeddings', True)
        self.embedding_cache_size = getattr(config, 'embedding_cache_size', 10000)

        # Initialize embedding cache if enabled
        if self.cache_embeddings:
            self.embedding_cache = {}  # text -> embedding
            self.cache_access_order = []  # for LRU eviction

        logger.info("NetworkX Vector Graph Store initialized with embedding support")

    def _get_cached_embedding(self, text: str) -> List[float]:
        """Get embedding from cache or generate new one"""
        if not self.cache_embeddings:
            # No caching, generate directly
            embedding = self.embedding_model.embed(text)
            return embedding if isinstance(embedding, list) else embedding.tolist()

        # Check cache first
        if text in self.embedding_cache:
            # Move to end for LRU
            self.cache_access_order.remove(text)
            self.cache_access_order.append(text)
            return self.embedding_cache[text]

        # Generate new embedding
        embedding = self.embedding_model.embed(text)
        embedding_list = embedding if isinstance(embedding, list) else embedding.tolist()

        # Add to cache with LRU eviction
        if len(self.embedding_cache) >= self.embedding_cache_size:
            # Remove oldest entry
            oldest_text = self.cache_access_order.pop(0)
            del self.embedding_cache[oldest_text]

        self.embedding_cache[text] = embedding_list
        self.cache_access_order.append(text)

        return embedding_list

    def generate_document_embedding(self, document: Document) -> List[float]:
        """Generate embedding for document content"""
        try:
            if not document.content:
                return []
            return self._get_cached_embedding(document.content)
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

            return self._get_cached_embedding(text_for_embedding)
        except Exception as e:
            logger.error(f"Failed to generate entity embedding: {e}")
            return []

    def add_document(self, document: Document) -> None:
        """Add document node with embedding"""
        # Generate embedding for document content
        embedding = self.generate_document_embedding(document)
        
        doc_node_id = f"doc_{document.id}"
        
        # Add document node to NetworkX graph with embedding
        self.graph.add_node(
            doc_node_id,
            node_type='Document',
            id_=document.id,
            content=document.content,
            metadata=json.dumps(document.metadata, ensure_ascii=False) if document.metadata else "{}",
            embedding=embedding,
            create_time=datetime.now().isoformat(),
            update_time=datetime.now().isoformat()
        )
        
        # Store document data separately for efficient retrieval (include embedding in metadata)
        metadata_with_embedding = document.metadata.copy() if document.metadata else {}
        if embedding:
            metadata_with_embedding['embedding'] = embedding
            
        self.documents[document.id] = {
            'content': document.content,
            'metadata': metadata_with_embedding,
            'create_time': datetime.now().isoformat(),
            'update_time': datetime.now().isoformat()
        }

    def add_graph_data(self, graph_data: GraphData, doc_id: str) -> None:
        """Add graph data (entities and relations) with entity embeddings"""
        doc_node_id = f"doc_{doc_id}"
        
        # Add entities with embeddings
        entity_node_mapping = {}  # entity_name -> node_id for this document
        
        for entity in graph_data.entities:
            entity_id = doc_id + '_' + entity['id']  # Prefix with doc_id to avoid conflicts
            entity_node_id = f"entity_{entity_id}"
            entity_type = entity.get('entity_type', 'Entity')
            
            # Generate embedding for entity
            entity_embedding = self.generate_entity_embedding(entity)
            
            # Add entity node with embedding
            self.graph.add_node(
                entity_node_id,
                node_type='Entity',
                entity_subtype=entity_type,
                id_=entity_id,
                entity_name=entity['entity_name'],
                entity_type=entity_type,
                document_id=doc_id,
                create_time=datetime.now().isoformat(),
                update_time=datetime.now().isoformat(),
                attributes=json.dumps(entity['attributes'], ensure_ascii=False) if entity.get('attributes') else "{}",
                embedding=entity_embedding
            )
            
            # Store mapping for relation creation
            entity_node_mapping[entity['entity_name']] = entity_node_id
            
            # Create Document-Entity relationship
            self.graph.add_edge(
                doc_node_id,
                entity_node_id,
                relation_type='MENTIONS',
                create_time=datetime.now().isoformat()
            )

        # Add relations between entities (same as parent class)
        for relation in graph_data.relations:
            if len(relation) >= 3:
                head_name, relation_type, tail_name = relation[0], relation[1], relation[2]
                
                # Find entity nodes by name within this document
                head_node_id = entity_node_mapping.get(head_name)
                tail_node_id = entity_node_mapping.get(tail_name)
                
                if head_node_id and tail_node_id:
                    # Add relationship edge
                    self.graph.add_edge(
                        head_node_id,
                        tail_node_id,
                        relation_type=relation_type,
                        document_id=doc_id,
                        create_time=datetime.now().isoformat()
                    )
                else:
                    logger.warning(f"Could not find entities for relation: {head_name} -> {tail_name}")

    def get_documents(self, ids: List[str]) -> List[Document]:
        """Get documents with embeddings"""
        documents = []

        for doc_id in ids:
            try:
                # Check if document exists
                if doc_id not in self.documents:
                    logger.warning(f"Document not found: {doc_id}")
                    continue

                doc_data = self.documents[doc_id]

                # Create document with embedding in metadata (already included from add_document)
                document = Document(
                    content=doc_data.get('content', ''),
                    id=doc_id,
                    metadata=doc_data.get('metadata', {})
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
        # Get entities for this document with embeddings
        entities = []
        entity_nodes = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if (node_data.get('node_type') == 'Entity' and 
                node_data.get('document_id') == doc_id):
                entity_nodes.append((node_id, node_data))
                
                entity = {
                    'id': node_data.get('id_', '').replace(f"{doc_id}_", ""),  # Remove doc_id prefix
                    'entity_name': node_data.get('entity_name', ''),
                    'entity_type': node_data.get('entity_type', ''),
                    'attributes': json.loads(node_data.get('attributes') or '{}')
                }
                
                # Add embedding to attributes if available
                if node_data.get('embedding'):
                    entity['attributes']['embedding'] = node_data['embedding']
                
                entities.append(entity)

        # Get relations between entities in this document (same as parent class)
        relations = []
        for edge in self.graph.edges(data=True):
            source, target, edge_data = edge
            
            # Check if this is an entity-entity relationship for this document
            if (edge_data.get('document_id') == doc_id and 
                edge_data.get('relation_type') != 'MENTIONS'):
                
                # Get entity names from nodes
                source_data = self.graph.nodes[source]
                target_data = self.graph.nodes[target]
                
                if (source_data.get('node_type') == 'Entity' and 
                    target_data.get('node_type') == 'Entity'):
                    relations.append([
                        source_data.get('entity_name', ''),
                        edge_data.get('relation_type', ''),
                        target_data.get('entity_name', '')
                    ])

        return GraphData(entities=entities, relations=relations, metadata={})

    def delete_all_Index(self, confirm: bool = False) -> bool:
        """Delete all documents and their graph data - implements abstract method"""
        return self.delete_all_index(confirm)

    def get_embeddings_by_type(self, node_type: str = 'Document') -> Dict[str, List[float]]:
        """Get all embeddings for nodes of a specific type
        
        Args:
            node_type: Type of nodes to get embeddings for ('Document' or 'Entity')
            
        Returns:
            Dictionary mapping node IDs to their embeddings
        """
        embeddings = {}
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('node_type') == node_type and node_data.get('embedding'):
                if node_type == 'Document':
                    # For documents, use the document ID
                    key = node_data.get('id_', node_id)
                else:
                    # For entities, use entity name or node ID
                    key = node_data.get('entity_name', node_id)
                embeddings[key] = node_data['embedding']
        
        return embeddings

    def similarity_search(self, query_embedding: List[float], node_type: str = 'Document',
                         top_k: int = None, similarity_threshold: float = None) -> List[Dict[str, Any]]:
        """Find most similar nodes based on embedding similarity

        Args:
            query_embedding: Query embedding vector
            node_type: Type of nodes to search ('Document' or 'Entity')
            top_k: Number of top results to return (uses config default if None)
            similarity_threshold: Minimum similarity threshold (uses config default if None)

        Returns:
            List of dictionaries with node information and similarity scores
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # Use config defaults if not specified
        if top_k is None:
            top_k = min(self.max_search_results, 100)  # Cap at 100 for safety
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold

        results = []
        query_embedding = np.array(query_embedding).reshape(1, -1)

        for node_id, node_data in self.graph.nodes(data=True):
            if (node_data.get('node_type') == node_type and
                node_data.get('embedding')):

                node_embedding = np.array(node_data['embedding']).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, node_embedding)[0][0]

                # Apply similarity threshold
                if similarity < similarity_threshold:
                    continue

                result = {
                    'node_id': node_id,
                    'similarity': float(similarity),
                    'data': node_data
                }

                if node_type == 'Document':
                    result['document_id'] = node_data.get('id_')
                    result['content'] = node_data.get('content', '')
                else:
                    result['entity_name'] = node_data.get('entity_name', '')
                    result['entity_type'] = node_data.get('entity_type', '')

                results.append(result)

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
